use std::path::Path;

use ort::session::Session;

use crate::inference::with_execution_mode;

use super::{EmbeddingModel, ExecutionMode};

impl EmbeddingModel {
    pub(super) fn build_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        Self::build_session_with_graph(model_path, mode, false)
    }

    pub(super) fn build_session_with_graph(
        model_path: &Path,
        mode: ExecutionMode,
        cuda_graph: bool,
    ) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(Self::intra_threads())?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder =
            if cuda_graph && matches!(mode, ExecutionMode::Cuda | ExecutionMode::CudaFast) {
                Self::with_cuda_graph_mode(builder)?
            } else {
                with_execution_mode(builder, mode)?
            };
        builder.commit_from_file(model_path)
    }

    #[cfg(feature = "cuda")]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        use ort::ep;

        Ok(builder.with_execution_providers([ep::CUDA::default()
            .with_device_id(0)
            .with_tf32(true)
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
            .with_conv_max_workspace(true)
            .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
            .with_prefer_nhwc(true)
            .with_cuda_graph(true)
            .build()
            .error_on_failure()])?)
    }

    #[cfg(not(feature = "cuda"))]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        with_execution_mode(builder, ExecutionMode::Cpu)
    }

    /// Intra-op thread count for the embedding ONNX sessions (tail / multimask / primary).
    ///
    /// diar-native patch, adopting the approach measured in upstream PR
    /// avencera/speakrs#6 by @ryoma0421: these sessions hardcoded `intra_threads(1)`,
    /// which leaves the embedding tail single-threaded and dominates wall time under
    /// `ExecutionMode::Cpu` (our CPU-only image tier). Under CUDA/CoreML the heavy ops
    /// are off-CPU, so the extra threads only serve small CPU-side glue nodes.
    ///
    /// The cap of 6 matches the already-shipped segmentation session builder
    /// (`SegmentationModel::build_session`), so the two model families now scale the
    /// same way. Sessions use independent thread pools and several are built per
    /// pipeline (tail, multimask, batched variants), but only one embedding session
    /// executes at a time within a request, so the concurrent thread demand is
    /// bounded by inflight-requests x 6 rather than sessions x 6. `SPEAKRS_INTRA_THREADS`
    /// exists to dial that down on small containers or high `DIAR_MAX_INFLIGHT` setups.
    fn intra_threads() -> usize {
        std::env::var("SPEAKRS_INTRA_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|count| count.get().min(6))
                    .unwrap_or(1)
            })
    }

    pub(super) fn build_fbank_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        // diar-native patch: default cap of 4 intra-op threads leaves fbank as ~76% of
        // CUDA E2E wall time on many-core hosts; allow override via SPEAKRS_FBANK_THREADS.
        let threads = std::env::var("SPEAKRS_FBANK_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|count| count.get().min(4))
                    .unwrap_or(1)
            });
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    pub(super) fn single_execution_mode(mode: ExecutionMode) -> ExecutionMode {
        match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => ExecutionMode::Cpu,
            _ => mode,
        }
    }

    pub(super) fn build_batched_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        Self::build_session(model_path, Self::single_execution_mode(mode))
    }
}
