use super::{BackendConfig, BackendError, RequestLayerImpl, build_bearer_headers, join_url};

pub(super) struct OpenaiEmbeddingsRequestLayer;

impl RequestLayerImpl for OpenaiEmbeddingsRequestLayer {
  fn build_url(&self, base_url: &str, _model: &str, _stream: bool) -> String {
    join_url(base_url, "/embeddings")
  }

  fn build_embedding_url(&self, base_url: &str, model: &str) -> Result<String, BackendError> {
    Ok(self.build_url(base_url, model, false))
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    build_bearer_headers(config, stream)
  }
}
