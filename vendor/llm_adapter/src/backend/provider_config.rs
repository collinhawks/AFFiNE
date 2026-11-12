use serde_json::{Value, json};

use super::{BackendProvider, CredentialPlan, TargetCompileError, canonicalize_endpoint};

#[derive(Clone, Copy)]
enum FieldKind {
  Text,
  Endpoint,
  Dialect,
  GoogleAuth,
}

struct ConfigField {
  name: &'static str,
  required: bool,
  kind: FieldKind,
}

const API_KEY: ConfigField = ConfigField {
  name: "apiKey",
  required: true,
  kind: FieldKind::Text,
};
const BASE_URL: ConfigField = ConfigField {
  name: "baseURL",
  required: false,
  kind: FieldKind::Endpoint,
};
const GOOGLE_FIELDS: &[ConfigField] = &[
  ConfigField {
    name: "project",
    required: true,
    kind: FieldKind::Text,
  },
  ConfigField {
    name: "location",
    required: true,
    kind: FieldKind::Text,
  },
  ConfigField {
    name: "googleAuthOptions",
    required: false,
    kind: FieldKind::GoogleAuth,
  },
];

pub(super) struct ProviderDefinition {
  pub(super) id: &'static str,
  fields: &'static [ConfigField],
  credential: CredentialKind,
  endpoint: EndpointTemplate,
}

enum CredentialKind {
  Field(&'static str),
  GoogleOAuth,
}

enum EndpointTemplate {
  Fixed(&'static str),
  Vertex(&'static str),
  Cloudflare,
}

impl BackendProvider {
  pub(super) fn definition(self) -> ProviderDefinition {
    use CredentialKind as Credential;
    use EndpointTemplate as Endpoint;
    let (id, fields, credential, endpoint): (_, &[_], _, _) = match self {
      Self::OpenAi => (
        "openai",
        &[
          API_KEY,
          ConfigField {
            name: "openaiDialect",
            required: false,
            kind: FieldKind::Dialect,
          },
        ],
        Credential::Field("apiKey"),
        Endpoint::Fixed("https://api.openai.com/v1"),
      ),
      Self::Anthropic => (
        "anthropic",
        &[API_KEY],
        Credential::Field("apiKey"),
        Endpoint::Fixed("https://api.anthropic.com"),
      ),
      Self::Gemini => (
        "gemini",
        &[API_KEY],
        Credential::Field("apiKey"),
        Endpoint::Fixed("https://generativelanguage.googleapis.com/v1beta"),
      ),
      Self::Fal => (
        "fal",
        &[API_KEY],
        Credential::Field("apiKey"),
        Endpoint::Fixed("https://fal.run"),
      ),
      Self::AnthropicVertex => (
        "anthropicVertex",
        GOOGLE_FIELDS,
        Credential::GoogleOAuth,
        Endpoint::Vertex("anthropic"),
      ),
      Self::GeminiVertex => (
        "geminiVertex",
        GOOGLE_FIELDS,
        Credential::GoogleOAuth,
        Endpoint::Vertex("google"),
      ),
      Self::CloudflareWorkersAi => (
        "cloudflareWorkersAi",
        &[
          ConfigField {
            name: "apiToken",
            required: true,
            kind: FieldKind::Text,
          },
          ConfigField {
            name: "accountId",
            required: true,
            kind: FieldKind::Text,
          },
        ],
        Credential::Field("apiToken"),
        Endpoint::Cloudflare,
      ),
      Self::Perplexity => (
        "perplexity",
        &[API_KEY],
        Credential::Field("apiKey"),
        Endpoint::Fixed("https://api.perplexity.ai"),
      ),
      Self::Openrouter => (
        "openrouter",
        &[API_KEY],
        Credential::Field("apiKey"),
        Endpoint::Fixed("https://openrouter.ai/api/v1"),
      ),
    };
    ProviderDefinition {
      id,
      fields,
      credential,
      endpoint,
    }
  }
}

fn field_schema(field: &ConfigField) -> Value {
  match field.kind {
    FieldKind::Text => json!({ "type": "string", "minLength": 1, "pattern": "\\S" }),
    FieldKind::Endpoint => {
      json!({ "type": "string", "pattern": r"^\s*[hH][tT][tT][pP][sS]?://[^/@?#]+(?:[/?#].*)?\s*$" })
    }
    FieldKind::Dialect => json!({ "type": "string", "enum": ["responses", "chat_completions"] }),
    FieldKind::GoogleAuth => json!({
      "type": "object", "properties": { "credentials": { "anyOf": [{
        "type": "object", "required": ["client_email", "private_key"],
        "properties": { "client_email": {"type":"string","minLength":1,"pattern":"\\S"},
          "private_key": {"type":"string","minLength":1,"pattern":"\\S"} }
      }, {"type":"null"}] } }
    }),
  }
}

pub fn backend_provider_config_schema(provider: BackendProvider) -> Value {
  let definition = provider.definition();
  let fields = definition.fields.iter().chain(std::iter::once(&BASE_URL));
  let mut properties = serde_json::Map::new();
  let mut required = Vec::new();
  for field in fields {
    let schema = field_schema(field);
    properties.insert(
      field.name.into(),
      if field.required {
        schema
      } else {
        json!({"anyOf":[schema, {"type":"null"}]})
      },
    );
    if field.required {
      required.push(field.name);
    }
  }
  json!({ "type": "object", "properties": properties, "required": required })
}

pub fn backend_providers_config_schema() -> Value {
  json!({ "oneOf": BackendProvider::ALL.into_iter().map(|provider| json!({
    "type": "object", "required": ["type", "config"],
    "properties": { "type": { "const": provider.as_str() }, "config": backend_provider_config_schema(provider) }
  })).collect::<Vec<_>>() })
}

pub fn validate_backend_config(provider: BackendProvider, config: &Value) -> Result<(), TargetCompileError> {
  let invalid = |message| TargetCompileError::InvalidConfig(message);
  let object = config
    .as_object()
    .ok_or_else(|| invalid("backend config must be an object".into()))?;
  let definition = provider.definition();
  for field in definition.fields.iter().chain(std::iter::once(&BASE_URL)) {
    let Some(value) = object.get(field.name) else {
      if field.required {
        return Err(invalid(format!("{} requires {}", definition.id, field.name)));
      }
      continue;
    };
    if !field.required && value.is_null() {
      continue;
    }
    match field.kind {
      FieldKind::Text if value.as_str().is_some_and(|text| !text.trim().is_empty()) => (),
      FieldKind::Endpoint => {
        let endpoint = value
          .as_str()
          .ok_or_else(|| invalid("baseURL must be a string".into()))?;
        canonicalize_endpoint(endpoint)?;
      }
      FieldKind::Dialect if matches!(value.as_str(), Some("responses" | "chat_completions")) => (),
      FieldKind::GoogleAuth if value.is_object() => {
        if let Some(credentials) = value.get("credentials").filter(|credentials| !credentials.is_null()) {
          for name in ["client_email", "private_key"] {
            if credentials
              .get(name)
              .and_then(Value::as_str)
              .is_none_or(|text| text.trim().is_empty())
            {
              return Err(invalid(format!("Google credentials require {name}")));
            }
          }
        }
      }
      _ => return Err(invalid(format!("invalid {} field {}", definition.id, field.name))),
    }
  }
  Ok(())
}

impl ProviderDefinition {
  pub(super) fn endpoint(&self, config: &Value) -> Result<String, TargetCompileError> {
    if let Some(value) = config.get("baseURL").filter(|value| !value.is_null()) {
      let endpoint = value
        .as_str()
        .ok_or_else(|| TargetCompileError::InvalidConfig("baseURL must be a string".into()))?;
      return canonicalize_endpoint(endpoint);
    }
    let endpoint = match self.endpoint {
      EndpointTemplate::Fixed(endpoint) => endpoint.to_string(),
      EndpointTemplate::Cloudflare => {
        let account = config
          .get("accountId")
          .and_then(Value::as_str)
          .filter(|value| !value.trim().is_empty())
          .ok_or_else(|| TargetCompileError::InvalidConfig("cloudflareWorkersAi requires accountId".into()))?;
        format!("https://api.cloudflare.com/client/v4/accounts/{account}/ai")
      }
      EndpointTemplate::Vertex(publisher) => {
        let project = config
          .get("project")
          .and_then(Value::as_str)
          .filter(|value| !value.trim().is_empty())
          .ok_or_else(|| TargetCompileError::InvalidConfig("Vertex requires project".into()))?;
        let location = config
          .get("location")
          .and_then(Value::as_str)
          .filter(|value| !value.trim().is_empty())
          .ok_or_else(|| TargetCompileError::InvalidConfig("Vertex requires location".into()))?;
        let host = if location == "global" {
          "aiplatform.googleapis.com".to_string()
        } else {
          format!("{location}-aiplatform.googleapis.com")
        };
        format!("https://{host}/v1/projects/{project}/locations/{location}/publishers/{publisher}")
      }
    };
    canonicalize_endpoint(&endpoint)
  }

  pub(super) fn credential_plan(&self, config: &Value) -> Result<CredentialPlan, TargetCompileError> {
    Ok(match self.credential {
      CredentialKind::Field(field) => CredentialPlan::Field { field: field.into() },
      CredentialKind::GoogleOAuth => CredentialPlan::GoogleOAuth {
        scopes: vec!["https://www.googleapis.com/auth/cloud-platform".into()],
        project: config
          .get("project")
          .and_then(Value::as_str)
          .filter(|value| !value.trim().is_empty())
          .ok_or_else(|| TargetCompileError::InvalidConfig("Vertex requires project".into()))?
          .into(),
        credentials: config
          .pointer("/googleAuthOptions/credentials")
          .filter(|credentials| !credentials.is_null())
          .cloned(),
      },
    })
  }
}

impl BackendProvider {
  pub fn uses_default_endpoint(self, config: &Value) -> bool {
    matches!(self.definition().endpoint, EndpointTemplate::Fixed(_)) && config.get("baseURL").is_none_or(Value::is_null)
  }
  pub fn endpoint(self, config: &Value) -> Result<String, TargetCompileError> {
    self.definition().endpoint(config)
  }
  pub fn credential_plan(self, config: &Value) -> Result<CredentialPlan, TargetCompileError> {
    self.definition().credential_plan(config)
  }
}
