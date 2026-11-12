use super::{
  BackendOperation, BackendRequestLayer, ModelRegistryVariant, ResolvedModelRegistryRoute, TargetCompileError,
  model_capabilities_upper_bound,
};

impl ModelRegistryVariant {
  pub fn matches_model(&self, model: &str) -> bool {
    self.raw_model_id == model
      || self.canonical_key == model
      || self.aliases.iter().any(|alias| alias == model)
      || self
        .legacy_aliases
        .as_ref()
        .is_some_and(|aliases| aliases.iter().any(|alias| alias == model))
  }

  pub fn operation_routes(
    &self,
    operation: BackendOperation,
  ) -> Result<Vec<ResolvedModelRegistryRoute>, TargetCompileError> {
    let route = self
      .route_overrides
      .as_ref()
      .and_then(|routes| routes.get(operation.route_key()));
    let layers = if let Some(route) = route {
      route.request_layers.clone()
    } else {
      let Some(layer) = self.request_layer.as_deref() else {
        return Ok(Vec::new());
      };
      vec![
        layer
          .parse::<BackendRequestLayer>()
          .map_err(|error| TargetCompileError::InvalidConfig(error.to_string()))?,
      ]
    };
    let capabilities = self
      .capabilities
      .iter()
      .cloned()
      .filter_map(|mut capability| {
        capability.output.retain(|output| match operation {
          BackendOperation::Chat => matches!(output.as_str(), "text" | "object"),
          _ => output == operation.route_key(),
        });
        if operation == BackendOperation::Structured {
          capability.attachments = capability.structured_attachments.clone().or(capability.attachments);
        }
        if operation == BackendOperation::Rerank {
          capability.input.retain(|input| input == "text");
          capability.attachments = None;
          capability.structured_attachments = None;
        }
        (!capability.input.is_empty() && !capability.output.is_empty()).then_some(capability)
      })
      .collect::<Vec<_>>();
    if capabilities.is_empty() {
      return Ok(Vec::new());
    }
    let capabilities = model_capabilities_upper_bound(
      &capabilities,
      &self.backend_kind,
      self.behavior_flags.as_deref().unwrap_or_default(),
    );
    Ok(
      layers
        .into_iter()
        .filter(|layer| layer.executable_protocol(operation).is_ok())
        .map(|layer| ResolvedModelRegistryRoute {
          request_layer: layer,
          wire_model: self.raw_model_id.clone(),
          capabilities: capabilities.clone(),
        })
        .collect(),
    )
  }
}
