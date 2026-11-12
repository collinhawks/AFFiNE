use serde_json::{Value, json};

use super::{
  BackendConfig, ChatCompletionsRequestLayer, RequestLayerImpl, ResponsesRequestLayer, build_bearer_headers, join_url,
};

pub(super) struct OpenrouterAnthropicRequestLayer;

impl RequestLayerImpl for OpenrouterAnthropicRequestLayer {
  fn build_url(&self, base_url: &str, _model: &str, _stream: bool) -> String {
    join_url(base_url, "/messages")
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    let mut headers = build_bearer_headers(config, stream);
    headers.push(("anthropic-version".into(), "2023-06-01".into()));
    headers.sort_by(|a, b| a.0.cmp(&b.0));
    headers
  }

  fn rewrite_body(&self, body: Value) -> Value {
    require_supported_parameters(body)
  }
}

pub(super) struct OpenrouterResponsesRequestLayer;

impl RequestLayerImpl for OpenrouterResponsesRequestLayer {
  fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String {
    ResponsesRequestLayer.build_url(base_url, model, stream)
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    ResponsesRequestLayer.build_headers(config, stream)
  }

  fn rewrite_body(&self, body: Value) -> Value {
    require_supported_parameters(body)
  }
}

pub(super) struct OpenrouterChatCompletionsRequestLayer;

impl RequestLayerImpl for OpenrouterChatCompletionsRequestLayer {
  fn build_url(&self, base_url: &str, model: &str, stream: bool) -> String {
    ChatCompletionsRequestLayer.build_url(base_url, model, stream)
  }

  fn build_headers(&self, config: &BackendConfig, stream: bool) -> Vec<(String, String)> {
    ChatCompletionsRequestLayer.build_headers(config, stream)
  }

  fn rewrite_body(&self, body: Value) -> Value {
    require_supported_parameters(body)
  }
}

fn require_supported_parameters(mut body: Value) -> Value {
  if let Value::Object(payload) = &mut body {
    payload.insert("provider".into(), json!({"require_parameters":true}));
  }
  body
}
