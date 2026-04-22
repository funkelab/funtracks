window.BENCHMARK_DATA = {
  "lastUpdate": 1776889941382,
  "repoUrl": "https://github.com/funkelab/funtracks",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "malinmayorc@janelia.hhmi.org",
            "name": "Caroline Malin-Mayor",
            "username": "cmalinmayor"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e7855a664fb707804557353b6d4e84bcfbc1c143",
          "message": "Merge pull request #201 from funkelab/fix-benchmark-on-main\n\nFix benchmark CI failing due to symlink conflict with mike docs deployment",
          "timestamp": "2026-04-22T16:31:47-04:00",
          "tree_id": "95a759f9f1ed087c3eac814bf231df68a8b17778",
          "url": "https://github.com/funkelab/funtracks/commit/e7855a664fb707804557353b6d4e84bcfbc1c143"
        },
        "date": 1776889939869,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/bench_candidate_graph.py::test_compute_graph_from_seg",
            "value": 0.3503964012938465,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 2.8539105889999945 sec\nrounds: 1"
          }
        ]
      }
    ]
  }
}