//! # Distributed Join Tests with Hive-Style Partitioning
//!
//! This module demonstrates two scenarios for distributed joins with Hive-style partitioning:
//!
//! ## Scenario 1: Default Task Estimator with Hive-Style Partitioning (OPTIMAL)
//! - Uses `FileScanConfigTaskEstimator` (default)
//! - Data: `testdata/join_test_hive/` with Hive-style partitioning (d_dkey=A/, f_dkey=A/, etc.)
//! - Workers: 2 workers, each processing 2 partitions (A,B on worker 0, C,D on worker 1)
//! - Result: Single-stage collocated join with NO network shuffles
//! - How: DataFusion recognizes matching partition keys and preserves file partitioning
//!
//! ## Scenario 2: Custom Partition Assignment Task Estimator (UNBALANCED MAPPING)
//! - Uses `PartitionAssignmentTaskEstimator` with `PartitionIsolatorExec::new_with_partition_groups()`
//! - Data: `testdata/join_test_hive/` with Hive-style partitioning
//! - Workers: 2 workers with UNBALANCED partition distribution
//!   - Worker 0: partition A (1 partition)
//!   - Worker 1: partitions B, C, D (3 partitions)
//! - Result: Single-stage collocated join with NO network shuffles
//! - How: Custom task estimator uses explicit partition groups `[[0], [1, 2, 3]]` to control
//!   which specific partitions are assigned to which workers, enabling unbalanced distribution
//!
//! ## Key Takeaway
//! To avoid network shuffles in distributed joins, use Hive-style partitioning with matching
//! partition keys on both tables. Custom task estimators can control how partitions are
//! distributed across workers, including unbalanced distributions.

#[cfg(all(feature = "integration", test))]
mod tests {
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::catalog::memory::DataSourceExec;
    use datafusion::config::ConfigOptions;
    use datafusion::datasource::physical_plan::FileScanConfig;
    use datafusion::error::DataFusionError;
    use datafusion::execution::{SessionState, SessionStateBuilder};
    use datafusion::physical_plan::{execute_stream, ExecutionPlan};
    use datafusion::prelude::CsvReadOptions;
    use datafusion_distributed::test_utils::localhost::start_localhost_context;
    use datafusion_distributed::{
        DistributedExt, DistributedSessionBuilderContext, PartitionIsolatorExec, TaskEstimation,
        TaskEstimator, display_plan_ascii,
    };
    use futures::TryStreamExt;
    use std::error::Error;
    use std::sync::Arc;

    async fn build_default_state(
        ctx: DistributedSessionBuilderContext,
    ) -> Result<SessionState, DataFusionError> {
        Ok(SessionStateBuilder::new()
            .with_runtime_env(ctx.runtime_env)
            .with_default_features()
            .build())
    }

    #[tokio::test]
    async fn test_join_with_default_task_estimator() -> Result<(), Box<dyn Error>> {
        // Start distributed context with 2 workers using DEFAULT task estimator
        // (no custom task estimator, just the built-in FileScanConfigTaskEstimator)
        let (ctx_distributed, _guard) = start_localhost_context(2, build_default_state).await;

        // Enable file partitioning preservation (from PR #19124)
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.preserve_file_partitions = 1;

        // Set target_partitions to 4 to create 4 file groups (one per Hive partition: A, B, C, D)
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Register dimension table with Hive-style partitioning
        let dim_options = CsvReadOptions::default()
            .table_partition_cols(vec![("d_dkey".to_string(), DataType::Utf8)]);
        ctx_distributed
            .register_csv("dim", "testdata/join_test_hive/dim", dim_options)
            .await?;

        // Register fact table with Hive-style partitioning
        // IMPORTANT: Declare sort order as (f_dkey, timestamp) following the pattern from
        // https://github.com/gene-bordegaray/datafusion/pull/3/files
        //
        // This is crucial for query optimization, especially for operations that
        // can benefit from knowing the global sort order.
        let fact_options = CsvReadOptions::default()
            .table_partition_cols(vec![("f_dkey".to_string(), DataType::Utf8)])
            .file_sort_order(vec![ // NGA: This sort order definition seems not working
                vec![
                    datafusion::prelude::col("f_dkey").sort(true, true), // ASC NULLS FIRST
                    datafusion::prelude::col("timestamp").sort(true, true), // ASC NULLS FIRST
                ]
            ]);
        ctx_distributed
            .register_csv("fact", "testdata/join_test_hive/fact", fact_options)
            .await?;

        // Create a join query
        let query = r#"
            SELECT 
                f.f_dkey,
                f.timestamp,
                f.value,
                d.env,
                d.service,
                d.host
            FROM dim d
            INNER JOIN fact f ON d.d_dkey = f.f_dkey
            ORDER BY f.f_dkey, f.timestamp
        "#;

        // Execute distributed query
        let df_distributed = ctx_distributed.sql(query).await?;
        let physical_distributed = df_distributed.create_physical_plan().await?;
        let physical_distributed_str = display_plan_ascii(physical_distributed.as_ref(), false);

        println!("\n=== DISTRIBUTED PLAN ===");
        println!("{}", physical_distributed_str);

        // Execute and collect distributed results
        let batches_distributed = execute_stream(physical_distributed.clone(), ctx_distributed.task_ctx())?
            .try_collect::<Vec<_>>()
            .await?;
        let result_distributed = pretty_format_batches(&batches_distributed)?;

        println!("\n=== DISTRIBUTED RESULTS ===");
        println!("{}", result_distributed);

        // Create a non-distributed context for comparison
        use datafusion::prelude::SessionContext;
        let ctx_non_distributed = SessionContext::new();

        // Enable file partitioning preservation in non-distributed context too
        ctx_non_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.preserve_file_partitions = 1;

        // Set target_partitions to 4 to match distributed context
        ctx_non_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Register the same tables in non-distributed context
        let dim_options_nd = CsvReadOptions::default()
            .table_partition_cols(vec![("d_dkey".to_string(), DataType::Utf8)]);
        ctx_non_distributed
            .register_csv("dim", "testdata/join_test_hive/dim", dim_options_nd)
            .await?;

        let fact_options_nd = CsvReadOptions::default()
            .table_partition_cols(vec![("f_dkey".to_string(), DataType::Utf8)])
            .file_sort_order(vec![
                vec![
                    datafusion::prelude::col("f_dkey").sort(true, true),
                    datafusion::prelude::col("timestamp").sort(true, true),
                ]
            ]);
        ctx_non_distributed
            .register_csv("fact", "testdata/join_test_hive/fact", fact_options_nd)
            .await?;

        // Execute the same query in non-distributed context
        let df_non_distributed = ctx_non_distributed.sql(query).await?;
        let physical_non_distributed = df_non_distributed.clone().create_physical_plan().await?;
        let physical_non_distributed_str = display_plan_ascii(physical_non_distributed.as_ref(), false);

        println!("\n=== NON-DISTRIBUTED PLAN ===");
        println!("{}", physical_non_distributed_str);

        let batches_non_distributed = df_non_distributed.collect().await?;
        let result_non_distributed = pretty_format_batches(&batches_non_distributed)?;

        println!("\n=== NON-DISTRIBUTED RESULTS ===");
        println!("{}", result_non_distributed);

        // Compare results: both should have the same data
        let total_rows_distributed: usize = batches_distributed.iter().map(|b| b.num_rows()).sum();
        let total_rows_non_distributed: usize = batches_non_distributed.iter().map(|b| b.num_rows()).sum();

        assert_eq!(
            total_rows_distributed, total_rows_non_distributed,
            "Row count mismatch: distributed={}, non-distributed={}",
            total_rows_distributed, total_rows_non_distributed
        );

        // Compare the actual data by converting to strings and comparing
        // Since both are sorted by (f_dkey, timestamp), they should be identical
        assert_eq!(
            result_distributed.to_string(),
            result_non_distributed.to_string(),
            "Results differ between distributed and non-distributed execution"
        );

        println!("\n✓ Verified: Distributed and non-distributed results are identical ({} rows)", total_rows_distributed);

        // Expected plan with f_dkey first (from fact table) and sorted fact table
        // The fact table is declared as sorted by (f_dkey, timestamp)
        // Still shows SortExec because the join doesn't preserve sort order
        let expected_plan = r#"┌───── DistributedExec ── Tasks: t0:[p0] 
│ SortPreservingMergeExec: [f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST]
│   [Stage 1] => NetworkCoalesceExec: output_partitions=4, input_tasks=2
└──────────────────────────────────────────────────
  ┌───── Stage 1 ── Tasks: t0:[p0..p1] t1:[p2..p3] 
  │ SortExec: expr=[f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST], preserve_partitioning=[true]
  │   ProjectionExec: expr=[f_dkey@5 as f_dkey, timestamp@3 as timestamp, value@4 as value, env@0 as env, service@1 as service, host@2 as host]
  │     HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@3, f_dkey@2)], projection=[env@0, service@1, host@2, timestamp@4, value@5, f_dkey@6]
  │       PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │         DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, service, host, d_dkey], file_type=csv, has_header=true
  │       PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │         DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data3.csv, /testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=C/data2.csv, /testdata/join_test_hive/fact/f_dkey=C/data.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
  └──────────────────────────────────────────────────
"#;

        println!("{}", physical_distributed_str);

        // Normalize paths for comparison
        let normalize_paths = |s: &str| -> String {
            s.lines()
                .map(|line| {
                    if line.contains("testdata/join_test_hive") {
                        let re = regex::Regex::new(r"[^,\s\[]*(/testdata/join_test_hive/[^,\s\]]+)").unwrap();
                        re.replace_all(line, "$1").to_string()
                    } else {
                        line.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        
        let normalized_expected = normalize_paths(&expected_plan);
        let normalized_actual = normalize_paths(&physical_distributed_str);
        
        assert_eq!(
            normalized_actual.trim(),
            normalized_expected.trim(),
            "\n\nActual plan does not match expected plan!\n\nExpected:\n{}\n\nActual:\n{}\n",
            normalized_expected,
            normalized_actual
        );

        // Expected non-distributed plan (with preserve_file_partitions enabled)
        let expected_non_distributed_plan = r#"SortPreservingMergeExec: [f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST]
  SortExec: expr=[f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST], preserve_partitioning=[true]
    ProjectionExec: expr=[f_dkey@5 as f_dkey, timestamp@3 as timestamp, value@4 as value, env@0 as env, service@1 as service, host@2 as host]
      HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@3, f_dkey@2)], projection=[env@0, service@1, host@2, timestamp@4, value@5, f_dkey@6]
        DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, service, host, d_dkey], file_type=csv, has_header=true
        DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data3.csv, /testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=C/data2.csv, /testdata/join_test_hive/fact/f_dkey=C/data.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
"#;

        let normalized_expected_nd = normalize_paths(&expected_non_distributed_plan);
        let normalized_actual_nd = normalize_paths(&physical_non_distributed_str);
        
        assert_eq!(
            normalized_actual_nd.trim(),
            normalized_expected_nd.trim(),
            "\n\nActual non-distributed plan does not match expected!\n\nExpected:\n{}\n\nActual:\n{}\n",
            normalized_expected_nd,
            normalized_actual_nd
        );

        Ok(())
    }

    /// Custom TaskEstimator that assigns partitions to workers based on a predefined list.
    ///
    /// This estimator takes a partition assignment like [[0], [1, 2, 3]] and ensures:
    /// - Worker 0 processes partition 0 (A)
    /// - Worker 1 processes partitions 1, 2, 3 (B, C, D)
    ///
    /// The key insight is that PartitionIsolatorExec distributes partitions evenly by default.
    /// To achieve custom distribution, we need to reorder the file groups so that when
    /// PartitionIsolatorExec divides them evenly, they end up on the correct workers.
    ///
    /// For example, with 4 partitions [A, B, C, D] and 2 workers:
    /// - Default: Worker 0 gets [A, B], Worker 1 gets [C, D]
    /// - Desired: Worker 0 gets [A], Worker 1 gets [B, C, D]
    ///
    /// We can't easily achieve [A] vs [B,C,D] split with PartitionIsolatorExec's even distribution.
    /// Instead, we'll pad the partitions to make the distribution work out.
    #[derive(Debug, Clone)]
    struct PartitionAssignmentTaskEstimator {
        /// List of partition indices per worker
        /// Example: vec![vec![0], vec![1, 2, 3]] means worker 0 gets partition 0, worker 1 gets partitions 1, 2, 3
        partition_groups: Vec<Vec<usize>>,
    }

    impl PartitionAssignmentTaskEstimator {
        fn new(partition_groups: Vec<Vec<usize>>) -> Self {
            Self { partition_groups }
        }
    }

    impl TaskEstimator for PartitionAssignmentTaskEstimator {
        fn estimate_tasks(
            &self,
            plan: &Arc<dyn ExecutionPlan>,
            _cfg: &ConfigOptions,
        ) -> Option<TaskEstimation> {
            // Try to downcast to DataSourceExec
            let dse: &DataSourceExec = plan.as_any().downcast_ref()?;
            let file_scan: &FileScanConfig = dse.data_source().as_any().downcast_ref()?;

            // Check if this is a Hive-partitioned table
            let first_path = file_scan
                .file_groups
                .first()?
                .iter()
                .next()?
                .object_meta
                .location
                .as_ref();

            // Only handle Hive-partitioned tables
            if !first_path.contains("join_test_hive/") {
                return None;
            }

            let task_count = self.partition_groups.len();
            let plan = Arc::clone(plan);
            
            // Use the NEW PartitionIsolatorExec::new_with_partition_groups() method
            // to create a PartitionIsolatorExec with explicit unbalanced partition assignment
            let plan = Arc::new(PartitionIsolatorExec::new_with_partition_groups(
                plan,
                self.partition_groups.clone(),
            ));

            Some(TaskEstimation {
                task_count,
                new_plan: Some(plan),
            })
        }
    }

    async fn build_partition_assignment_state(
        ctx: DistributedSessionBuilderContext,
        partition_groups: Vec<Vec<usize>>,
    ) -> Result<SessionState, DataFusionError> {
        Ok(SessionStateBuilder::new()
            .with_runtime_env(ctx.runtime_env)
            .with_default_features()
            .with_distributed_task_estimator(PartitionAssignmentTaskEstimator::new(
                partition_groups,
            ))
            .build())
    }

    #[tokio::test]
    async fn test_join_with_partition_assignment_task_estimator() -> Result<(), Box<dyn Error>> {
        // Start distributed context with 2 workers
        // Worker 0 will process partition A (1 partition)
        // Worker 1 will process partitions B, C, D (3 partitions)
        // This demonstrates UNBALANCED custom partition-to-worker assignment
        let partition_groups = vec![
            vec![0],        // Worker 0: partition A only
            vec![1, 2, 3],  // Worker 1: partitions B, C, D
        ];
        let (ctx_distributed, _guard) = start_localhost_context(
            partition_groups.len(),
            move |ctx| build_partition_assignment_state(ctx, partition_groups.clone()),
        )
        .await;

        // Enable file partitioning preservation
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.preserve_file_partitions = 1;

        // Set target_partitions to 4 to create 4 file groups (one per Hive partition: A, B, C, D)
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Register dimension table with Hive-style partitioning
        let dim_options = CsvReadOptions::default()
            .table_partition_cols(vec![("d_dkey".to_string(), DataType::Utf8)]);
        ctx_distributed
            .register_csv("dim", "testdata/join_test_hive/dim", dim_options)
            .await?;

        // Register fact table with Hive-style partitioning
        let fact_options = CsvReadOptions::default()
            .table_partition_cols(vec![("f_dkey".to_string(), DataType::Utf8)]);
        ctx_distributed
            .register_csv("fact", "testdata/join_test_hive/fact", fact_options)
            .await?;

        // Create a join query
        let query = r#"
            SELECT 
                d.d_dkey,
                d.env,
                d.service,
                d.host,
                f.timestamp,
                f.value
            FROM dim d
            INNER JOIN fact f ON d.d_dkey = f.f_dkey
            ORDER BY d.d_dkey, f.timestamp
        "#;

        // Execute distributed query
        let df_distributed = ctx_distributed.sql(query).await?;
        let physical_distributed = df_distributed.create_physical_plan().await?;
        let physical_distributed_str = display_plan_ascii(physical_distributed.as_ref(), false);

        // Execute and collect results
        let batches_distributed = execute_stream(physical_distributed.clone(), ctx_distributed.task_ctx())?
            .try_collect::<Vec<_>>()
            .await?;
        let _result_distributed = pretty_format_batches(&batches_distributed)?;

        // NGA: PPLAN LOOKS GOOD BUT WRONG RESULTS
        //      NEED TO USE DistributedTaskContext instead
        // Expected plan with UNBALANCED partition assignment:
        // - Worker 0 (t0): Processes partition A (1 partition)
        // - Worker 1 (t1): Processes partitions B, C, D (3 partitions)
        //
        // The PartitionIsolatorExec shows: t0:[p0,__,__,__] t1:[__,p0,p1,p2]
        // This means:
        // - Task 0 gets partition 0 (A) from the input
        // - Task 1 gets partitions 1, 2, 3 (B, C, D) from the input
        //
        // This is a SINGLE-STAGE collocated join with NO network shuffles!
        let expected_plan = r#"┌───── DistributedExec ── Tasks: t0:[p0] 
│ SortPreservingMergeExec: [d_dkey@0 ASC NULLS LAST, timestamp@4 ASC NULLS LAST]
│   [Stage 1] => NetworkCoalesceExec: output_partitions=2, input_tasks=2
└──────────────────────────────────────────────────
  ┌───── Stage 1 ── Tasks: t0:[p0] t1:[p1] 
  │ SortExec: expr=[d_dkey@0 ASC NULLS LAST, timestamp@4 ASC NULLS LAST], preserve_partitioning=[true]
  │   ProjectionExec: expr=[d_dkey@3 as d_dkey, env@0 as env, service@1 as service, host@2 as host, timestamp@4 as timestamp, value@5 as value]
  │     HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@3, f_dkey@2)], projection=[env@0, service@1, host@2, d_dkey@3, timestamp@4, value@5]
  │       PartitionIsolatorExec: t0:[p0,__,__,__] t1:[__,p0,p1,p2] 
  │         DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, service, host, d_dkey], file_type=csv, has_header=true
  │       PartitionIsolatorExec: t0:[p0,__,__,__] t1:[__,p0,p1,p2] 
  │         DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data3.csv, /testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=C/data2.csv, /testdata/join_test_hive/fact/f_dkey=C/data.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
  └──────────────────────────────────────────────────
"#;



        println!("{}", physical_distributed_str);

        // Normalize paths for comparison
        let normalize_paths = |s: &str| -> String {
            s.lines()
                .map(|line| {
                    if line.contains("testdata/join_test_hive") {
                        let re = regex::Regex::new(r"[^,\s\[]*(/testdata/join_test_hive/[^,\s\]]+)").unwrap();
                        re.replace_all(line, "$1").to_string()
                    } else {
                        line.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        
        let normalized_expected = normalize_paths(&expected_plan);
        let normalized_actual = normalize_paths(&physical_distributed_str);
        
        assert_eq!(
            normalized_actual.trim(),
            normalized_expected.trim(),
            "\n\nActual plan does not match expected plan!\n\nExpected:\n{}\n\nActual:\n{}\n",
            normalized_expected,
            normalized_actual
        );

        Ok(())
    }
}
