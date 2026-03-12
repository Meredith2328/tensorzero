use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use futures::{FutureExt, TryFutureExt};
use sqlx::PgPool;
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::task::JoinSet;

use crate::config::BatchWritesConfig;
use crate::db::BatchWriterHandle;
use crate::db::batching::process_bounded_channel_with_capacity_and_timeout;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{
    ChatInferenceDatabaseInsert, JsonInferenceDatabaseInsert, StoredModelInference,
};

use super::inference_queries::{
    build_insert_chat_inference_data_query, build_insert_chat_inferences_query,
    build_insert_json_inference_data_query, build_insert_json_inferences_query,
};
use super::model_inferences::{
    build_insert_model_inference_data_query, build_insert_model_inferences_query,
};

/// An async job to be executed by a flush worker.
type FlushJob = Pin<Box<dyn Future<Output = ()> + Send>>;

/// A `PostgresBatchSender` is used to submit entries to the batch writer, which aggregates
/// and submits them to Postgres on a schedule defined by a `BatchWritesConfig`.
///
/// Architecture (two-stage pipeline):
/// 1. **Bounded input channels**: Rows are sent via `try_send`. When full, rows are dropped and logged.
/// 2. **Accumulator tasks**: One per table type, accumulates rows into batches (max_rows or timeout).
/// 3. **Concurrent flush pool**: Ready batches are submitted as jobs to N flush workers that
///    execute bulk INSERTs in parallel against the PgPool.
///
/// When a `PostgresBatchSender` is dropped, the batch writer will finish
/// processing all outstanding batches once all senders are dropped.
#[derive(Debug)]
pub struct PostgresBatchSender {
    chat_inferences: Sender<ChatInferenceDatabaseInsert>,
    json_inferences: Sender<JsonInferenceDatabaseInsert>,
    model_inferences: Sender<StoredModelInference>,
    pub writer_handle: BatchWriterHandle,
}

impl PostgresBatchSender {
    pub fn new(
        pool: PgPool,
        config: BatchWritesConfig,
        channel_capacity: usize,
    ) -> Result<Self, Error> {
        // We call `tokio::task::block_in_place` during shutdown to wait for outstanding
        // batch writes to finish. This does not work on the CurrentThread runtime,
        // so we fail here rather than panicking at shutdown.
        if Handle::current().runtime_flavor() == RuntimeFlavor::CurrentThread
            && !config.__force_allow_embedded_batch_writes
        {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Cannot use Postgres batching with the CurrentThread Tokio runtime"
                    .to_string(),
            }));
        }

        let (chat_tx, chat_rx) = mpsc::channel(channel_capacity);
        let (json_tx, json_rx) = mpsc::channel(channel_capacity);
        let (model_tx, model_rx) = mpsc::channel(channel_capacity);

        let writer = PostgresBatchWriter {
            chat_inferences_rx: chat_rx,
            json_inferences_rx: json_rx,
            model_inferences_rx: model_rx,
        };

        let handle = tokio::runtime::Handle::current();
        // We use `spawn_blocking` to ensure that when the runtime shuts down, it waits for this task to complete.
        let writer_handle = tokio::task::spawn_blocking(move || {
            handle.block_on(async move {
                tracing::debug!("Postgres batch write handler started");
                writer.process(pool, config).await;
                tracing::info!("Postgres batch write handler finished");
            });
        });

        Ok(Self {
            chat_inferences: chat_tx,
            json_inferences: json_tx,
            model_inferences: model_tx,
            writer_handle: writer_handle.map_err(|e| format!("{e:?}")).boxed().shared(),
        })
    }

    pub fn send_chat_inferences(&self, rows: &[ChatInferenceDatabaseInsert]) {
        for row in rows {
            if let Err(e) = self.chat_inferences.try_send(row.clone()) {
                tracing::error!(
                    "Postgres batch channel full — dropping chat inference record. \
                     Increase `write_queue_capacity` or check Postgres performance. Error: {e}"
                );
            }
        }
    }

    pub fn send_json_inferences(&self, rows: &[JsonInferenceDatabaseInsert]) {
        for row in rows {
            if let Err(e) = self.json_inferences.try_send(row.clone()) {
                tracing::error!(
                    "Postgres batch channel full — dropping json inference record. \
                     Increase `write_queue_capacity` or check Postgres performance. Error: {e}"
                );
            }
        }
    }

    pub fn send_model_inferences(&self, rows: &[StoredModelInference]) {
        for row in rows {
            if let Err(e) = self.model_inferences.try_send(row.clone()) {
                tracing::error!(
                    "Postgres batch channel full — dropping model inference record. \
                     Increase `write_queue_capacity` or check Postgres performance. Error: {e}"
                );
            }
        }
    }
}

struct PostgresBatchWriter {
    chat_inferences_rx: Receiver<ChatInferenceDatabaseInsert>,
    json_inferences_rx: Receiver<JsonInferenceDatabaseInsert>,
    model_inferences_rx: Receiver<StoredModelInference>,
}

/// A concurrent pool of flush workers that execute bulk INSERT jobs.
///
/// Jobs are submitted via a bounded channel. Workers pick up jobs and execute them
/// against the PgPool. When the sender is dropped, workers drain remaining jobs and exit.
/// Call `drain()` after dropping all senders to await worker completion.
struct FlushPool {
    flush_sender: Sender<FlushJob>,
    worker_handles: JoinSet<()>,
}

impl FlushPool {
    fn new(num_workers: usize, flush_queue_capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel::<FlushJob>(flush_queue_capacity);
        let rx = std::sync::Arc::new(tokio::sync::Mutex::new(rx));

        let mut worker_handles = JoinSet::new();
        for worker_id in 0..num_workers {
            let rx = rx.clone();
            worker_handles.spawn(async move {
                loop {
                    let job = {
                        let mut rx = rx.lock().await;
                        rx.recv().await
                    };
                    match job {
                        Some(job) => job.await,
                        None => {
                            tracing::debug!(worker_id, "Flush pool worker shutting down");
                            break;
                        }
                    }
                }
            });
        }

        Self {
            flush_sender: tx,
            worker_handles,
        }
    }

    /// Close the flush queue and await all workers to drain remaining jobs.
    async fn drain(mut self) {
        // Drop the sender so workers see channel closure after draining remaining jobs.
        drop(self.flush_sender);
        while let Some(result) = self.worker_handles.join_next().await {
            if let Err(e) = result {
                tracing::error!("Flush pool worker panicked: {e}");
            }
        }
    }
}

impl PostgresBatchWriter {
    async fn process(self, pool: PgPool, config: BatchWritesConfig) {
        let batch_timeout = Duration::from_millis(config.flush_interval_ms);
        let max_rows = config.max_rows_postgres.unwrap_or(config.max_rows);
        let flush_concurrency = config.flush_concurrency;

        // The flush queue capacity is small — just enough to keep workers busy.
        // Backpressure from a full flush queue propagates to the accumulator,
        // which stops draining the input channel, which fills up and triggers drops.
        let flush_queue_capacity = flush_concurrency * 2;
        let flush_pool = FlushPool::new(flush_concurrency, flush_queue_capacity);

        let mut accumulator_set = JoinSet::new();

        // Chat inferences accumulator
        {
            let pool = pool.clone();
            let flush_sender = flush_pool.flush_sender.clone();
            let channel = self.chat_inferences_rx;
            accumulator_set.spawn(async move {
                process_bounded_channel_with_capacity_and_timeout(
                    channel,
                    max_rows,
                    batch_timeout,
                    move |buffer| {
                        let pool = pool.clone();
                        let flush_sender = flush_sender.clone();
                        async move {
                            let job: FlushJob = Box::pin(flush_chat_inferences(pool, buffer));
                            if let Err(e) = flush_sender.send(job).await {
                                tracing::error!("Flush pool closed unexpectedly: {e}");
                            }
                            Vec::with_capacity(max_rows)
                        }
                    },
                )
                .await;
            });
        }

        // JSON inferences accumulator
        {
            let pool = pool.clone();
            let flush_sender = flush_pool.flush_sender.clone();
            let channel = self.json_inferences_rx;
            accumulator_set.spawn(async move {
                process_bounded_channel_with_capacity_and_timeout(
                    channel,
                    max_rows,
                    batch_timeout,
                    move |buffer| {
                        let pool = pool.clone();
                        let flush_sender = flush_sender.clone();
                        async move {
                            let job: FlushJob = Box::pin(flush_json_inferences(pool, buffer));
                            if let Err(e) = flush_sender.send(job).await {
                                tracing::error!("Flush pool closed unexpectedly: {e}");
                            }
                            Vec::with_capacity(max_rows)
                        }
                    },
                )
                .await;
            });
        }

        // Model inferences accumulator
        {
            let flush_sender = flush_pool.flush_sender.clone();
            let channel = self.model_inferences_rx;
            accumulator_set.spawn(async move {
                process_bounded_channel_with_capacity_and_timeout(
                    channel,
                    max_rows,
                    batch_timeout,
                    move |buffer| {
                        let pool = pool.clone();
                        let flush_sender = flush_sender.clone();
                        async move {
                            let job: FlushJob = Box::pin(flush_model_inferences(pool, buffer));
                            if let Err(e) = flush_sender.send(job).await {
                                tracing::error!("Flush pool closed unexpectedly: {e}");
                            }
                            Vec::with_capacity(max_rows)
                        }
                    },
                )
                .await;
            });
        }

        // Wait for all accumulators to finish (they finish when input channels close)
        while let Some(result) = accumulator_set.join_next().await {
            if let Err(e) = result {
                tracing::error!("Error in Postgres batch accumulator: {e}");
            }
        }

        // All accumulators done — drain the flush pool, awaiting all in-flight INSERTs.
        // This is critical for shutdown: without this, spawned flush workers could be
        // abandoned before completing their final INSERTs, causing data loss.
        flush_pool.drain().await;
    }
}

/// Execute both chat inference INSERTs (metadata + data) concurrently.
async fn flush_chat_inferences(pool: PgPool, buffer: Vec<ChatInferenceDatabaseInsert>) {
    let row_count = buffer.len();
    let metadata_future = async {
        match build_insert_chat_inferences_query(&buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), &pool, "chat_inferences", row_count)
                        .await
                {
                    tracing::error!("Error writing chat inferences to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building chat inferences query: {e}");
            }
        }
    };
    let data_future = async {
        match build_insert_chat_inference_data_query(&buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), &pool, "chat_inference_data", row_count)
                        .await
                {
                    tracing::error!("Error writing chat inference data to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building chat inference data query: {e}");
            }
        }
    };
    tokio::join!(metadata_future, data_future);
}

/// Execute both JSON inference INSERTs (metadata + data) concurrently.
async fn flush_json_inferences(pool: PgPool, buffer: Vec<JsonInferenceDatabaseInsert>) {
    let row_count = buffer.len();
    let metadata_future = async {
        match build_insert_json_inferences_query(&buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), &pool, "json_inferences", row_count)
                        .await
                {
                    tracing::error!("Error writing json inferences to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building json inferences query: {e}");
            }
        }
    };
    let data_future = async {
        match build_insert_json_inference_data_query(&buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), &pool, "json_inference_data", row_count)
                        .await
                {
                    tracing::error!("Error writing json inference data to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building json inference data query: {e}");
            }
        }
    };
    tokio::join!(metadata_future, data_future);
}

/// Execute both model inference INSERTs (metadata + data) concurrently.
async fn flush_model_inferences(pool: PgPool, buffer: Vec<StoredModelInference>) {
    let row_count = buffer.len();
    let metadata_future = async {
        match build_insert_model_inferences_query(&buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), &pool, "model_inferences", row_count)
                        .await
                {
                    tracing::error!("Error writing model inferences to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building model inferences query: {e}");
            }
        }
    };
    let data_future = async {
        match build_insert_model_inference_data_query(&buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), &pool, "model_inference_data", row_count)
                        .await
                {
                    tracing::error!("Error writing model inference data to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building model inference data query: {e}");
            }
        }
    };
    tokio::join!(metadata_future, data_future);
}
