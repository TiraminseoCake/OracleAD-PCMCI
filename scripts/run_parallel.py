#!/usr/bin/env python3
"""GPU-sharded parallel launcher for PICAAD.

One JOB = one (entity, seed). Each JOB launches a fresh `python main.py`
subprocess pinned to a single GPU via CUDA_VISIBLE_DEVICES. When a GPU
frees up, the next pending job is dispatched to it.

Usage:
    python scripts/run_parallel.py --dataset SWaT --gpus 0,1,2,3 --out_dir results/parallel/swat_$(date +%Y%m%d-%H%M%S)
    python scripts/run_parallel.py --dataset PSM  --gpus 0,1,2,3
    python scripts/run_parallel.py --dataset SMD  --gpus 0,1,2,3 --out_dir results/parallel/smd_$(date +%Y%m%d-%H%M%S)

After all jobs finish, best-epoch aggregation runs automatically:
    -> {out_dir}/summary_all_epochs.csv
    -> {out_dir}/summary_best_epoch.csv
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# SMD 28 entities (machine-1-{1..8}, machine-2-{1..9}, machine-3-{1..11})
SMD_ENTITIES = [
    f'machine-{g}-{i}'
    for g, n in [(1, 8), (2, 9), (3, 11)]
    for i in range(1, n + 1)
]

DATASETS = {
    'PSM':  ('scripts/configs/psm.yaml',  ['PSM']),
    'SMD':  ('scripts/configs/smd.yaml',  SMD_ENTITIES),
    'SWaT': ('scripts/configs/swat.yaml', ['swat']),
}


def build_cmd(cfg_yaml, entity, seed, run_dir, epochs):
    return [
        PY, '-u', 'main.py',
        '--cfg', cfg_yaml,
        'SEEDS', f'[{seed}]',
        'DATA.ENTITIES', entity,
        'RESULT_DIR', run_dir,
        'RESULT_DIR_LITERAL', 'True',
        'SOLVER.MAX_EPOCH', str(epochs),
    ]


_WARMUP_SCRIPT = (
    'import sys; sys.path.insert(0, ".");'
    'from config import get_cfg_defaults;'
    'from datasets.build import load_entity;'
    'from model.build import build_causal_prior_cached;'
    'cfg = get_cfg_defaults();'
    'cfg.merge_from_file(sys.argv[1]);'
    'cfg.merge_from_list(["DATA.ENTITIES", sys.argv[2]]);'
    'entity = load_entity(cfg, sys.argv[2]);'
    'build_causal_prior_cached(cfg, entity.train_z, entity.name)'
)


def warmup_priors(cfg_yaml, entities, threads_per_warmup):
    """Serially precompute the PCMCI+ prior for each unique entity so that
    seed-parallel training subprocesses all cache-hit instead of duplicating
    the (identical) computation with different seeds.

    Serial by design: one PCMCI+ call at a time, but each call is free to use
    many BLAS threads for speed. Total warmup wall-clock is
    ~len(entities) * (single-entity PCMCI+ time).
    """
    if not entities:
        return
    print(f'\n[warmup] priming prior cache for {len(entities)} unique entities '
          f'({threads_per_warmup} threads per call, serial)', flush=True)
    t0 = time.time()
    for i, ent in enumerate(entities):
        cmd = [PY, '-c', _WARMUP_SCRIPT, cfg_yaml, ent]
        env = dict(os.environ,
                    OMP_NUM_THREADS=str(threads_per_warmup),
                    MKL_NUM_THREADS=str(threads_per_warmup),
                    OPENBLAS_NUM_THREADS=str(threads_per_warmup),
                    NUMEXPR_NUM_THREADS=str(threads_per_warmup))
        t_ent = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        dt = int(time.time() - t_ent)
        status = 'OK' if r.returncode == 0 else f'FAIL(exit={r.returncode})'
        print(f'  [{i + 1}/{len(entities)}] {ent}: {status} ({dt}s)', flush=True)
    total = int(time.time() - t0)
    print(f'[warmup] done in {total}s\n', flush=True)


def _fmt_eta(remaining, per_job):
    if per_job is None or per_job <= 0:
        return '?'
    secs = int(remaining * per_job)
    h, m = divmod(secs // 60, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m'


def main():
    ap = argparse.ArgumentParser(description='PICAAD GPU-parallel launcher')
    ap.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    ap.add_argument('--gpus', required=True, help='e.g. 0,1,2,3')
    ap.add_argument('--seeds', default='0,1,2,3', help='comma-separated (default: 0,1,2,3)')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--out_dir', default='',
                    help='Root output dir. Empty -> results/parallel/{dataset}_{timestamp}')
    ap.add_argument('--poll_seconds', type=int, default=5)
    ap.add_argument('--threads_per_job', type=int, default=4,
                    help='OMP/MKL/OpenBLAS thread cap per subprocess. Keep '
                         'threads_per_job * (num GPUs) <= num CPU cores to '
                         'avoid CPU contention during PCMCI+ computation.')
    ap.add_argument('--no_aggregate', action='store_true',
                    help='Skip automatic best-epoch aggregation after all jobs finish')
    ap.add_argument('--no_warmup', action='store_true',
                    help='Skip serial per-entity prior warmup. Each training '
                         'subprocess will then compute PCMCI+ independently '
                         '(wasteful when seeds run in parallel for the same '
                         'entity).')
    ap.add_argument('--warmup_threads', type=int, default=16,
                    help='BLAS threads for each serial warmup call (default: 16).')
    ap.add_argument('--entities', default='',
                    help='Comma-separated entity subset. Overrides the '
                         'built-in DATASETS registry. Useful for restarting '
                         'a partial run without re-doing completed entities.')
    a = ap.parse_args()

    cfg_yaml, entities = DATASETS[a.dataset]
    if a.entities:
        entities = [e.strip() for e in a.entities.split(',') if e.strip()]
    gpus = [g.strip() for g in a.gpus.split(',') if g.strip()]
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]

    if not a.out_dir:
        ts = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        a.out_dir = f'results/parallel/{a.dataset.lower()}_{ts}'

    out_root = Path(a.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # One job = one (entity, seed)
    jobs = [(e, s) for e in entities for s in seeds]
    total = len(jobs)

    print(f'[parallel] dataset={a.dataset} entities={len(entities)} seeds={seeds} '
          f'-> {total} jobs', flush=True)
    print(f'[parallel] gpus={gpus} epochs={a.epochs} out_root={out_root}', flush=True)

    # Prior warmup: fill the on-disk PCMCI+ cache per unique entity BEFORE
    # spawning any training subprocess. Otherwise all seed-parallel workers
    # for the same entity duplicate the identical prior computation and
    # thrash the CPU with unbounded BLAS threads.
    if not a.no_warmup:
        unique_entities = list(dict.fromkeys(entities))
        warmup_priors(cfg_yaml, unique_entities, a.warmup_threads)

    running = {}       # gpu -> (Popen, ent, seed, log_file, start_time)
    free = list(gpus)
    pending = list(jobs)
    done = 0
    failed = []
    completion_times = []
    t_start = time.time()

    while pending or running:
        # Launch jobs to any free GPUs
        while pending and free:
            ent, seed = pending.pop(0)
            gpu = free.pop(0)
            run_dir = out_root / ent / f'seed{seed}'
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / 'run.log'

            cmd = build_cmd(cfg_yaml, ent, seed, str(run_dir), a.epochs)
            # Limit BLAS thread count so N parallel processes don't spawn
            # N*num_cores threads (which cause massive CPU contention during
            # the CPU-heavy PCMCI+ prior computation stage).
            env = dict(os.environ,
                        CUDA_VISIBLE_DEVICES=gpu,
                        PYTHONUNBUFFERED='1',
                        OMP_NUM_THREADS=str(a.threads_per_job),
                        MKL_NUM_THREADS=str(a.threads_per_job),
                        OPENBLAS_NUM_THREADS=str(a.threads_per_job),
                        NUMEXPR_NUM_THREADS=str(a.threads_per_job))

            log_file = open(log_path, 'w')
            p = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                  stdout=log_file, stderr=subprocess.STDOUT)
            running[gpu] = (p, ent, seed, log_file, time.time())

            elapsed = int(time.time() - t_start)
            in_flight = done + len(running)
            avg = (sum(completion_times) / len(completion_times)) if completion_times else None
            eta = _fmt_eta((total - done - len(running)) / max(len(gpus), 1), avg)
            print(f'[launch {in_flight}/{total}] {ent} seed{seed} gpu{gpu} '
                  f'(elapsed {elapsed}s, eta {eta})', flush=True)

        # Poll running jobs
        time.sleep(a.poll_seconds)
        for gpu in list(running.keys()):
            p, ent, seed, log_file, t_job = running[gpu]
            if p.poll() is not None:
                log_file.close()
                done += 1
                completion_times.append(time.time() - t_job)
                elapsed = int(time.time() - t_start)
                if p.returncode != 0:
                    failed.append((ent, seed, p.returncode))
                    print(f'[FAIL {done}/{total}] {ent} seed{seed} gpu{gpu} '
                          f'exit={p.returncode} (elapsed {elapsed}s) log={out_root / ent / f"seed{seed}" / "run.log"}',
                          flush=True)
                else:
                    print(f'[done {done}/{total}] {ent} seed{seed} gpu{gpu} exit=0 '
                          f'(job {int(completion_times[-1])}s, total elapsed {elapsed}s)',
                          flush=True)
                del running[gpu]
                free.append(gpu)

    elapsed = int(time.time() - t_start)
    print(f'\n[parallel] ALL DONE in {elapsed}s. '
          f'success={done - len(failed)}/{total} failed={len(failed)}', flush=True)
    if failed:
        print('  FAILED jobs:', flush=True)
        for ent, seed, rc in failed:
            print(f'    {ent} seed{seed} exit={rc}', flush=True)

    if not a.no_aggregate:
        print(f'\n[parallel] running best-epoch aggregation ...', flush=True)
        subprocess.run([PY, 'scripts/aggregate_best_epoch.py',
                         '--run_dir', str(out_root)],
                        cwd=str(ROOT), check=False)

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
