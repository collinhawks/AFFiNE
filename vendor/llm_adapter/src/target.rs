use std::{
  fmt,
  net::{IpAddr, SocketAddr},
};

use thiserror::Error;
use url::Url;
use zeroize::Zeroizing;

use crate::{
  backend::{BackendConfig, BackendError, BackendRequestLayer},
  router::{ExecutablePreparedRoute, ExecutableProtocol, ExecutableRequest},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendOperation {
  Chat,
  Structured,
  Embedding,
  Rerank,
  Image,
}
impl BackendOperation {
  pub fn route_key(self) -> &'static str {
    match self {
      Self::Chat => "text",
      Self::Structured => "structured",
      Self::Embedding => "embedding",
      Self::Rerank => "rerank",
      Self::Image => "image",
    }
  }
}
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiDialect {
  Responses,
  ChatCompletions,
}

pub struct BackendCredential(Zeroizing<String>);
impl BackendCredential {
  #[must_use]
  pub fn new(value: String) -> Self {
    Self(Zeroizing::new(value))
  }
}
impl fmt::Debug for BackendCredential {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("BackendCredential([REDACTED])")
  }
}
#[derive(Debug, Clone)]
pub struct PreparedBackendTarget {
  model: String,
  layer: BackendRequestLayer,
  protocol: ExecutableProtocol,
  endpoint: String,
  timeout_ms: Option<u64>,
  egress_policy: EgressPolicy,
}
impl PreparedBackendTarget {
  pub fn wire_model(&self) -> &str {
    &self.model
  }
  pub fn request_layer(&self) -> BackendRequestLayer {
    self.layer
  }
  pub fn endpoint(&self) -> &str {
    &self.endpoint
  }
  pub fn bind(
    &self,
    credential: BackendCredential,
    request: ExecutableRequest,
  ) -> Result<ExecutablePreparedRoute, BackendError> {
    ExecutablePreparedRoute::new(
      self.protocol,
      self.model.clone(),
      BackendConfig {
        base_url: self.endpoint.clone(),
        auth_token: credential.0.as_str().into(),
        request_layer: Some(self.layer),
        headers: Default::default(),
        no_streaming: false,
        timeout_ms: self.timeout_ms,
        egress_policy: self.egress_policy,
      },
      request,
    )
  }
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum EgressPolicy {
  #[default]
  PublicOnly,
  AllowPrivate,
}
#[derive(Debug, Error)]
pub enum TargetCompileError {
  #[error("invalid target configuration: {0}")]
  InvalidConfig(String),
  #[error("model id is required")]
  EmptyModel,
  #[error("model id exceeds 512 bytes")]
  ModelTooLong,
  #[error("invalid endpoint: {0}")]
  InvalidEndpoint(String),
  #[error("endpoint host could not be resolved")]
  UnresolvedEndpoint(#[source] std::io::Error),
  #[error("endpoint resolves to a disallowed network address")]
  DisallowedAddress,
}
pub fn canonicalize_endpoint(value: &str) -> Result<String, TargetCompileError> {
  let mut url = Url::parse(value.trim()).map_err(|e| TargetCompileError::InvalidEndpoint(e.to_string()))?;
  if !matches!(url.scheme(), "http" | "https")
    || url.host_str().is_none()
    || !url.username().is_empty()
    || url.password().is_some()
  {
    return Err(TargetCompileError::InvalidEndpoint(
      "endpoint must be an HTTP(S) URL without user info".into(),
    ));
  }
  url.set_query(None);
  url.set_fragment(None);
  let path = url.path().trim_end_matches('/').to_string();
  url.set_path(if path.is_empty() { "/" } else { &path });
  Ok(url.to_string().trim_end_matches('/').to_string())
}
pub fn prepare_backend_target(
  layer: BackendRequestLayer,
  operation: BackendOperation,
  wire_model: String,
  endpoint: &str,
  timeout_ms: Option<u64>,
  egress_policy: EgressPolicy,
) -> Result<PreparedBackendTarget, TargetCompileError> {
  if wire_model.trim().is_empty() {
    return Err(TargetCompileError::EmptyModel);
  }
  if wire_model.len() > 512 {
    return Err(TargetCompileError::ModelTooLong);
  }
  let protocol = layer
    .executable_protocol(operation)
    .map_err(|e| TargetCompileError::InvalidConfig(e.to_string()))?;
  Ok(PreparedBackendTarget {
    model: wire_model,
    layer,
    protocol,
    endpoint: canonicalize_endpoint(endpoint)?,
    timeout_ms,
    egress_policy,
  })
}
impl EgressPolicy {
  pub fn resolve(self, url: &str) -> Result<Vec<SocketAddr>, TargetCompileError> {
    let url = Url::parse(url).map_err(|error| TargetCompileError::InvalidEndpoint(error.to_string()))?;
    let addresses = url
      .socket_addrs(|| None)
      .map_err(TargetCompileError::UnresolvedEndpoint)?;
    if addresses.is_empty() {
      return Err(TargetCompileError::UnresolvedEndpoint(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "DNS returned no addresses",
      )));
    }
    if self == Self::PublicOnly && addresses.iter().any(|address| !is_public(address.ip())) {
      return Err(TargetCompileError::DisallowedAddress);
    }
    Ok(addresses)
  }
}

fn is_public(ip: IpAddr) -> bool {
  match ip {
    IpAddr::V4(ip) => {
      !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_unspecified()
        || ip.is_multicast())
    }
    IpAddr::V6(ip) => {
      !(ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (ip.segments()[0] & 0xfe00) == 0xfc00
        || (ip.segments()[0] & 0xffc0) == 0xfe80
        || (ip.segments()[0] == 0x2001 && ip.segments()[1] == 0x0db8))
    }
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::{
    backend::{ApiShape, BackendProvider, CredentialPlan, validate_backend_config},
    core::{default_model_registry_variants, resolve_model_registry_variant},
    test_support::sample_request,
  };

  #[test]
  fn keeps_opaque_model_id_and_canonicalizes_custom_endpoint() {
    for model in [String::new(), " ".into(), "m".repeat(513)] {
      assert!(
        prepare_backend_target(
          BackendRequestLayer::Responses,
          BackendOperation::Chat,
          model,
          "https://example.com",
          None,
          EgressPolicy::PublicOnly
        )
        .is_err()
      );
    }
    let target = prepare_backend_target(
      BackendRequestLayer::OpenrouterResponses,
      BackendOperation::Chat,
      " vendor/model:latest ".into(),
      "https://gateway.example/api/?ignored=1#fragment",
      Some(1000),
      EgressPolicy::PublicOnly,
    )
    .unwrap();
    assert_eq!(target.wire_model(), " vendor/model:latest ");
    assert_eq!(target.endpoint(), "https://gateway.example/api");
    let route = target
      .bind(
        BackendCredential::new("secret".into()),
        ExecutableRequest::Chat(sample_request()),
      )
      .unwrap();
    assert_eq!(route.model, " vendor/model:latest ");
    assert_eq!(
      route.config.request_layer,
      Some(BackendRequestLayer::OpenrouterResponses)
    );
    assert_eq!(route.config.timeout_ms, Some(1000));
    assert!(
      target
        .bind(
          BackendCredential::new("secret".into()),
          ExecutableRequest::Embedding(crate::core::EmbeddingRequest {
            model: "ignored".into(),
            inputs: vec!["text".into()],
            dimensions: None,
            task_type: None
          })
        )
        .is_err()
    );
    assert!(!format!("{target:?}").contains("secret"));
  }

  #[test]
  fn compiles_provider_and_model_bindings() {
    for operation in [
      BackendOperation::Chat,
      BackendOperation::Structured,
      BackendOperation::Embedding,
      BackendOperation::Rerank,
      BackendOperation::Image,
    ] {
      for dialect in [
        None,
        Some(OpenAiDialect::Responses),
        Some(OpenAiDialect::ChatCompletions),
      ] {
        let layer = BackendProvider::OpenAi
          .default_request_layer(operation, dialect)
          .unwrap();
        let target = prepare_backend_target(
          layer,
          operation,
          "opaque/wire:model".into(),
          &BackendProvider::OpenAi.endpoint(&json!({})).unwrap(),
          None,
          EgressPolicy::PublicOnly,
        )
        .unwrap();
        assert_eq!(target.wire_model(), "opaque/wire:model");
        assert_eq!(target.endpoint(), "https://api.openai.com/v1");
        if matches!(operation, BackendOperation::Chat | BackendOperation::Structured) {
          assert_eq!(
            layer,
            if dialect == Some(OpenAiDialect::ChatCompletions) {
              BackendRequestLayer::ChatCompletions
            } else {
              BackendRequestLayer::Responses
            }
          );
          assert!(
            BackendProvider::OpenAi
              .configured_layer_matches(&json!({"openaiDialect":dialect}), layer, operation)
              .unwrap()
          );
          if dialect.is_some() {
            let other = if layer == BackendRequestLayer::Responses {
              BackendRequestLayer::ChatCompletions
            } else {
              BackendRequestLayer::Responses
            };
            assert!(
              !BackendProvider::OpenAi
                .configured_layer_matches(&json!({"openaiDialect":dialect}), other, operation)
                .unwrap()
            );
          }
        }
      }
    }
    for (provider, operation, layer) in [
      (
        BackendProvider::Anthropic,
        BackendOperation::Chat,
        BackendRequestLayer::Anthropic,
      ),
      (
        BackendProvider::AnthropicVertex,
        BackendOperation::Chat,
        BackendRequestLayer::VertexAnthropic,
      ),
      (
        BackendProvider::Gemini,
        BackendOperation::Chat,
        BackendRequestLayer::GeminiApi,
      ),
      (
        BackendProvider::GeminiVertex,
        BackendOperation::Chat,
        BackendRequestLayer::GeminiVertex,
      ),
      (
        BackendProvider::CloudflareWorkersAi,
        BackendOperation::Rerank,
        BackendRequestLayer::CloudflareWorkersAi,
      ),
      (BackendProvider::Fal, BackendOperation::Image, BackendRequestLayer::Fal),
      (
        BackendProvider::Perplexity,
        BackendOperation::Chat,
        BackendRequestLayer::PerplexitySonar,
      ),
      (
        BackendProvider::Openrouter,
        BackendOperation::Chat,
        BackendRequestLayer::OpenrouterResponses,
      ),
    ] {
      assert_eq!(provider.default_request_layer(operation, None).unwrap(), layer);
      assert!(
        provider
          .default_request_layer(operation, Some(OpenAiDialect::Responses))
          .is_err()
      );
    }
    let variants = default_model_registry_variants();
    for (provider, model, shapes) in [
      (BackendProvider::OpenAi, "gpt-6-luna", &[ApiShape::Responses][..]),
      (BackendProvider::OpenAi, "o1", &[ApiShape::ChatCompletions][..]),
      (
        BackendProvider::Openrouter,
        "gpt-6-luna",
        &[ApiShape::Responses, ApiShape::ChatCompletions, ApiShape::Anthropic][..],
      ),
      (
        BackendProvider::Openrouter,
        "claude-sonnet-4-6",
        &[ApiShape::Responses, ApiShape::ChatCompletions, ApiShape::Anthropic][..],
      ),
    ] {
      let variant = variants
        .iter()
        .find(|variant| provider.matches_registry_backend(&variant.backend_kind) && variant.matches_model(model))
        .unwrap();
      let unique = resolve_model_registry_variant(&variants, Some(&variant.backend_kind), model)
        .unwrap()
        .unwrap()
        .0;
      assert_eq!(unique, variant);
      let routes = variant.operation_routes(BackendOperation::Chat).unwrap();
      assert_eq!(routes.len(), shapes.len());
      for route in routes {
        assert!(shapes.contains(&route.request_layer.api_shape()));
      }
      let structured = variant.operation_routes(BackendOperation::Structured).unwrap();
      assert!(
        structured
          .iter()
          .all(|route| route.request_layer.api_shape() != ApiShape::Anthropic)
      );
    }
    let rerank = variants
      .iter()
      .find(|variant| variant.backend_kind == "openrouter" && variant.matches_model("gpt-4.1"))
      .unwrap();
    let routes = rerank.operation_routes(BackendOperation::Rerank).unwrap();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].request_layer, BackendRequestLayer::OpenrouterChatCompletions);
    assert!(
      routes[0]
        .capabilities
        .iter()
        .all(|capability| capability.features.is_empty())
    );
  }

  #[test]
  fn validates_backend_configuration_and_credentials() {
    for (provider, config, endpoint) in [
      (
        BackendProvider::OpenAi,
        json!({"apiKey":"secret", "baseURL":null, "openaiDialect":null}),
        "https://api.openai.com/v1",
      ),
      (
        BackendProvider::Anthropic,
        json!({"apiKey":"secret"}),
        "https://api.anthropic.com",
      ),
      (
        BackendProvider::Openrouter,
        json!({"apiKey":"secret"}),
        "https://openrouter.ai/api/v1",
      ),
      (
        BackendProvider::GeminiVertex,
        json!({"project":"test", "location":"global", "googleAuthOptions":null}),
        "https://aiplatform.googleapis.com/v1/projects/test/locations/global/publishers/google",
      ),
      (
        BackendProvider::CloudflareWorkersAi,
        json!({"accountId":"account", "apiToken":"secret"}),
        "https://api.cloudflare.com/client/v4/accounts/account/ai",
      ),
    ] {
      assert_eq!(provider.endpoint(&config).unwrap(), endpoint);
      validate_backend_config(provider, &config).unwrap();
      #[cfg(feature = "schema")]
      assert!(jsonschema::is_valid(
        &crate::backend::backend_provider_config_schema(provider),
        &config
      ));
      let credential = provider.credential_plan(&config).unwrap();
      assert!(!format!("{credential:?}").contains("secret"));
      if provider == BackendProvider::OpenAi {
        assert!(matches!(credential, CredentialPlan::Field { field } if field == "apiKey"));
      }
    }
    for config in [
      json!({}),
      json!({"apiKey":" "}),
      json!({"apiKey":2}),
      json!({"apiKey":"secret","baseURL":"file:///tmp/model"}),
    ] {
      assert!(validate_backend_config(BackendProvider::Openrouter, &config).is_err());
    }
    for endpoint in ["file:///tmp/model", "https://user:secret@example.com", "not-a-url"] {
      assert!(canonicalize_endpoint(endpoint).is_err());
    }
    assert!(BackendProvider::OpenAi.uses_default_endpoint(&json!({})));
    assert!(BackendProvider::OpenAi.uses_default_endpoint(&json!({"baseURL":null})));
    assert!(!BackendProvider::OpenAi.uses_default_endpoint(&json!({"baseURL":"https://api.openai.com/v1"})));
    assert!(!BackendProvider::GeminiVertex.uses_default_endpoint(&json!({"project":"p", "location":"global"})));
    assert!(!BackendProvider::CloudflareWorkersAi.uses_default_endpoint(&json!({"accountId":"a"})));
    assert_eq!(
      BackendProvider::Anthropic.endpoint(&json!({})).unwrap(),
      "https://api.anthropic.com"
    );
    assert_eq!(
      BackendProvider::Openrouter
        .endpoint(&json!({"baseURL":"https://gateway.example/api/"}))
        .unwrap(),
      "https://gateway.example/api"
    );
    assert!(BackendProvider::GeminiVertex.endpoint(&json!({})).is_err());
    assert!(BackendProvider::CloudflareWorkersAi.endpoint(&json!({})).is_err());
    assert!(BackendProvider::GeminiVertex.credential_plan(&json!({})).is_err());
    assert!(validate_backend_config(BackendProvider::Openrouter, &json!({"apiKey":" "})).is_err());
    assert!(
      prepare_backend_target(
        BackendRequestLayer::Anthropic,
        BackendOperation::Structured,
        "model".into(),
        "https://example.com",
        None,
        EgressPolicy::PublicOnly
      )
      .is_err()
    );
  }

  #[test]
  fn rejects_private_egress() {
    for endpoint in ["http://127.0.0.1", "http://[::1]", "http://192.168.1.1"] {
      assert!(matches!(
        EgressPolicy::PublicOnly.resolve(endpoint),
        Err(TargetCompileError::DisallowedAddress)
      ));
      assert!(EgressPolicy::AllowPrivate.resolve(endpoint).is_ok());
    }
  }
}
