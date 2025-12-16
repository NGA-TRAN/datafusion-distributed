## Test

- test file: `tests/join_time_agg_space_agg.rs`
- test function: `test_join_with_time_agg_then_space_agg_and_default_task_estimator`
   - See settings there
   - Using PR #19304: `gene.bordegaray/2025/12/hash_partitioning_satisfies_subset`
   - PR link: https://github.com/apache/datafusion/pull/19304

### Test data

The test uses Hive-style partitioned data located in `testdata/join_test_hive/`:


***Data Partitions**

```
    testdata/join_test_hive/
    ├── dim/
    │   ├── d_dkey=A/
    │   │   └── data.csv
    │   ├── d_dkey=B/
    │   │   └── data.csv
    │   ├── d_dkey=C/
    │   │   └── data.csv
    │   └── d_dkey=D/
    │       └── data.csv
    └── fact/
        ├── f_dkey=A/
        │   └── data.csv
        ├── f_dkey=B/
        │   ├── data.csv
        │   ├── data2.csv
        │   └── data3.csv
        ├── f_dkey=C/
        │   ├── data.csv
        │   └── data2.csv
        └── f_dkey=D/
            └── data.csv
```

- dim is partitioned by d_dkey  (join key of the query)
- fact is partitioned by f_dkey (join key of the query) and sorted by (f_dkey, timestamp)


### Query and its plan

```SQL
SELECT env, time_bin, AVG(max_bin_value) AS avg_max_value
FROM
(
    SELECT  f_dkey, 
            date_bin(INTERVAL '30 seconds', timestamp) AS time_bin,
            MAX(env) AS env,
            MAX(value) AS max_bin_value
    FROM
        (
        SELECT 
            f.f_dkey,
            d.env,
            d.service,
            d.host,
            f.timestamp,
            f.value
        FROM dim d
        INNER JOIN fact f ON d.d_dkey = f.f_dkey
        WHERE service = 'log'
        ) AS j
    GROUP BY f_dkey, time_bin
) AS a
GROUP BY env, time_bin
ORDER BY env, time_bin


┌───── DistributedExec ── Tasks: t0:[p0] 
│ SortPreservingMergeExec: [env@0 ASC NULLS LAST, time_bin@1 ASC NULLS LAST]
│   SortExec: expr=[env@0 ASC NULLS LAST, time_bin@1 ASC NULLS LAST], preserve_partitioning=[true]
│     ProjectionExec: expr=[env@0 as env, time_bin@1 as time_bin, avg(a.max_bin_value)@2 as avg_max_value]
│       AggregateExec: mode=FinalPartitioned, gby=[env@0 as env, time_bin@1 as time_bin], aggr=[avg(a.max_bin_value)]
│         [Stage 1] => NetworkShuffleExec: output_partitions=4, input_tasks=2
└──────────────────────────────────────────────────
  ┌───── Stage 1 ── Tasks: t0:[p0..p3] t1:[p0..p3] 
  │ CoalesceBatchesExec: target_batch_size=8192
  │   RepartitionExec: partitioning=Hash([env@0, time_bin@1], 4), input_partitions=2
  │     AggregateExec: mode=Partial, gby=[env@1 as env, time_bin@0 as time_bin], aggr=[avg(a.max_bin_value)]
  │       ProjectionExec: expr=[date_bin(IntervalMonthDayNano("IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }"),j.timestamp)@1 as time_bin, max(j.env)@2 as env, max(j.value)@3 as max_bin_value]
  │         AggregateExec: mode=SinglePartitioned, gby=[f_dkey@0 as f_dkey, date_bin(IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }, timestamp@2) as date_bin(IntervalMonthDayNano("IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }"),j.timestamp)], aggr=[max(j.env), max(j.value)], ordering_mode=Sorted
  │           ProjectionExec: expr=[f_dkey@3 as f_dkey, env@0 as env, timestamp@1 as timestamp, value@2 as value]
  │             HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@1, f_dkey@2)], projection=[env@0, timestamp@2, value@3, f_dkey@4]
  │               FilterExec: service@1 = log, projection=[env@0, d_dkey@2]
  │                 PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │                   DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, service, d_dkey], file_type=csv, has_header=true
  │               PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │                 DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data3.csv, /testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=C/data2.csv, /testdata/join_test_hive/fact/f_dkey=C/data.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
  └──────────────────────────────────────────────────

```


![Distributed POC Diagram](./distributedPOC.png)

## TODOs

0. MUST: Verify why there is no Dynamic Filtering? CVS files?
1. MUST: Distributed dim & fact files correctly for partitioned hash join 
2. MUST: Build custom TaskEstimator 
3. OPTIONAL: Make space aggregation happen in stage1
4. INVESTIGATE: The `repartition_subset_satisfactions` flag from PR #19304 doesn't seem to work in distributed context
   - Tests show 2-stage plans with repartitioning instead of expected single-stage `mode=SinglePartitioned`
   - May need to investigate if distributed planner is interfering with this optimization 

