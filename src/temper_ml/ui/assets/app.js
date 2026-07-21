"use strict";

const csrfToken = document.querySelector('meta[name="temper-csrf"]').content;
const rawEvidence = document.querySelector("#raw-evidence");
const notice = document.querySelector("#notice");
let workspace = null;
let activeStage = "setup";
let latestComparison = null;
let blindAliases = [];
let preferredReviewIdentity = null;
let renderedCleanupPlanKey = null;
let operationPoll = null;
let clientOperationTimer = null;
let clientOperationStartedAt = null;
let clientOperationAction = null;

const viewMeta = {
  setup: ["Workspace / 01", "Overview", "Project state, evidence, and the next bounded action."],
  data: ["Workspace / 02", "Data", "Import, inspect, and freeze deterministic training examples."],
  recipe: ["Workspace / 03", "Recipes", "Resolve visible candidate settings against one target."],
  run: ["Workspace / 04", "Runs", "Follow runtime evidence from launch through output integrity."],
  evaluate: ["Workspace / 05", "Evaluate", "Compare synchronized outputs and record honest review evidence."],
  use: ["Workspace / 06", "Local use", "Select one verified output for focused local work and export."],
  storage: ["Workspace / 07", "Storage & replay", "Inspect local bytes, preview cleanup consequences, and reproduce transparently."],
};

const stageLabels = {
  setup: "project setup",
  data: "data import",
  recipe: "recipe resolution",
  run: "candidate runs",
  evaluate: "evaluation",
  use: "local use",
  storage: "storage and replay",
};

const actions = {
  setup: () => post("/api/v1/setup", {
    mode: value("training-mode"),
    model_source: value("model-source") || undefined,
    tokenizer_source: value("tokenizer-source") || undefined,
    display_name: value("model-display-name"),
    revision: value("model-revision"),
    target: value("execution-target"),
  }),
  import: () => importDataset(),
  resolve: () => post("/api/v1/candidates/resolve", { options: recipeOptions() }),
  launch: () => post("/api/v1/runs/launch", {}),
  cancel: () => post("/api/v1/runs/cancel", {}),
  compare: () => post("/api/v1/playground/compare", {
    prompt: value("playground-prompt"),
    maximum_tokens: numberValue("maximum-tokens"),
    seed: numberValue("inference-seed"),
  }),
  "solo-review": () => post("/api/v1/playground/reviews/solo", reviewBody()),
  "blind-prepare": () => post("/api/v1/playground/reviews/blind/prepare", {}),
  "blind-seal": () => post("/api/v1/playground/reviews/blind/seal", blindReviewBody()),
  "blind-reveal": () => post("/api/v1/playground/reviews/blind/reveal", {}),
  evaluate: () => post("/api/v1/evaluation/run", {}),
  capture: () => {
    const review = selectedCompletedReview();
    return post("/api/v1/evaluation/capture", {
      review_identity: review.reference.identity,
      suite_kind: "development",
    });
  },
  select: () => post("/api/v1/decisions", {
    candidate_key: value("candidate-selection"),
    status: "selected",
  }),
  focused: () => post("/api/v1/local-use/focused", {
    candidate_key: value("candidate-selection"),
    prompt: value("focused-prompt"),
    maximum_tokens: 64,
    seed: 17,
    save: true,
  }),
  batch: () => post("/api/v1/local-use/batch", {
    candidate_key: value("candidate-selection"),
    prompts: value("batch-prompts").split(/\r?\n/).map((item) => item.trim()).filter(Boolean),
    maximum_tokens: 64,
    seed: 17,
    save: false,
  }),
  export: () => post("/api/v1/exports", {
    candidate_key: value("candidate-selection"),
  }),
  "cleanup-preview": () => {
    if (workspace?.retention?.active_plan) throw new Error("cleanup_plan_active");
    const entryIds = selectedStorageEntryIds();
    if (!entryIds.length) throw new Error("cleanup_selection_required");
    return post("/api/v1/storage/cleanup/preview", { entry_ids: entryIds });
  },
  "cleanup-execute": () => {
    const plan = workspace?.retention?.active_plan;
    if (!plan?.plan_id) throw new Error("cleanup_plan_required");
    const entryIds = selectedStorageEntryIds().sort();
    const plannedEntryIds = [...(plan.selected_entry_ids || [])].sort();
    if (!sameStrings(entryIds, plannedEntryIds)) {
      throw new Error("cleanup_selection_plan_mismatch");
    }
    if (!document.querySelector("#cleanup-confirmation").checked) {
      throw new Error("cleanup_confirmation_required");
    }
    return post("/api/v1/storage/cleanup/execute", {
      plan_id: plan.plan_id,
      entry_ids: entryIds,
      confirm: true,
    });
  },
  "replay-plan": () => {
    if (workspace?.reproduction?.active_plan) throw new Error("replay_plan_active");
    return post("/api/v1/replays/plan", {
      candidate_key: value("replay-candidate"),
      mode: value("replay-mode"),
    });
  },
  "replay-execute": () => {
    const plan = workspace?.reproduction?.active_plan;
    if (!plan?.plan_id || !plan?.run_id) throw new Error("replay_plan_required");
    const candidateKey = value("replay-candidate");
    const mode = value("replay-mode");
    if (candidateKey !== plan.candidate_key || mode !== plan.mode) {
      throw new Error("replay_controls_plan_mismatch");
    }
    return post("/api/v1/replays/execute", {
      plan_id: plan.plan_id,
      run_id: plan.run_id,
      candidate_key: candidateKey,
      mode,
    });
  },
};

document.querySelectorAll("[data-stage]").forEach((button) => {
  button.addEventListener("click", () => showStage(button.dataset.stage));
});

document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", async () => {
    if (button.dataset.targetStage) {
      showStage(button.dataset.targetStage);
      return;
    }
    const action = actions[button.dataset.action];
    if (!action) return;
    setBusy(true, button.dataset.action);
    try {
      const payload = await action();
      consumeAction(button.dataset.action, payload.result);
      workspace = payload.workspace;
      renderWorkspace();
      showNotice(successMessage(button.dataset.action), false);
    } catch (error) {
      showNotice(actionableError(error), true);
    } finally {
      setBusy(false);
      if (workspace) renderOperation();
    }
  });
});

document.querySelector("#hf-source-mode").addEventListener("change", syncHuggingFaceSourceMode);
document.querySelector("#training-mode").addEventListener("change", syncSetupControls);
syncHuggingFaceSourceMode();
syncSetupControls();

loadWorkspace();

async function loadWorkspace() {
  try {
    const response = await fetch("/api/v1/workspace", {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error?.code || "workspace_unavailable");
    workspace = payload.data;
    renderWorkspace();
  } catch (error) {
    showNotice(`Workspace unavailable: ${error.message}`, true);
  }
}

async function post(path, body) {
  const cleanBody = Object.fromEntries(Object.entries(body).filter(([, item]) => item !== undefined));
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      "X-Temper-CSRF": csrfToken,
    },
    body: JSON.stringify(cleanBody),
  });
  const payload = await response.json();
  showRawEvidence(payload);
  if (!response.ok || !payload.ok) throw apiError(payload);
  return payload.data;
}

async function importDataset() {
  const sourceKind = value("dataset-source-kind");
  if (sourceKind === "fixture") {
    return post("/api/v1/dataset/import", { format: "fixture" });
  }
  const options = datasetOptions();
  if (sourceKind === "hugging_face") {
    return post("/api/v1/dataset/import", { format: "hugging_face", options });
  }
  if (sourceKind === "paste") {
    return post("/api/v1/dataset/import", {
      format: value("dataset-format"), source: value("dataset-source"), options,
    });
  }
  const file = document.querySelector("#dataset-file").files[0];
  if (!file) throw new Error("Choose a local JSON, JSONL, or CSV file first.");
  const query = new URLSearchParams({
    format: value("dataset-format"), options: JSON.stringify(options),
  });
  return uploadDatasetFile(`/api/v1/dataset/import-file?${query}`, file);
}

function datasetOptions() {
  const huggingFaceMode = value("hf-source-mode");
  const options = {
    dataset_url: value("hf-url"),
    hugging_face_source_mode: huggingFaceMode,
    context_field: value("field-context"), completion_field: value("field-completion"),
    cot_field: value("field-cot") || undefined, output_field: value("field-output") || undefined,
    renderer: value("dataset-renderer"), row_limit: numberValue("dataset-row-limit"),
    maximum_tokens: numberValue("dataset-max-tokens"),
    maximum_characters: optionalNumberValue("dataset-max-characters"),
    train_weight: numberValue("train-weight"), validation_weight: numberValue("validation-weight"),
  };
  if (huggingFaceMode === "repository_file") options.file_path = value("hf-file") || undefined;
  else {
    options.config = value("hf-config");
    options.split = value("hf-split");
  }
  return options;
}

function uploadDatasetFile(path, file) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", path);
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.setRequestHeader("Accept", "application/json");
    request.setRequestHeader("X-Temper-CSRF", csrfToken);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      renderClientOperation(`Uploading local file · ${Math.round((event.loaded / event.total) * 100)}%`);
    });
    request.addEventListener("load", () => {
      let payload;
      try {
        payload = JSON.parse(request.responseText);
      } catch (_error) {
        reject(new Error("invalid_server_response"));
        return;
      }
      showRawEvidence(payload);
      if (request.status < 200 || request.status >= 300 || !payload.ok) {
        reject(apiError(payload));
        return;
      }
      resolve(payload.data);
    });
    request.addEventListener("error", () => reject(new Error("file_upload_failed")));
    request.send(file);
  });
}

function syncHuggingFaceSourceMode() {
  const mode = value("hf-source-mode");
  document.querySelectorAll("[data-hf-mode]").forEach((field) => {
    const active = field.dataset.hfMode === mode;
    field.hidden = !active;
    field.querySelectorAll("input, select").forEach((control) => {
      control.disabled = !active;
    });
  });
}

function syncSetupControls() {
  const real = value("training-mode") === "real_local";
  for (const id of ["execution-target", "model-source", "tokenizer-source", "model-display-name", "model-revision"]) {
    document.getElementById(id).disabled = !real;
  }
}

function recipeOptions() {
  return {
    sequence_length: numberValue("recipe-sequence-length"),
    training_steps: numberValue("recipe-training-steps"), rank: numberValue("recipe-rank"),
    alpha: numberValue("recipe-alpha"), target_modules: value("recipe-target-modules"),
    precision: value("recipe-precision"), checkpoint_cadence: 1,
  };
}

function apiError(payload) {
  const error = new Error(payload.error?.code || "request_failed");
  error.details = payload.error?.details || {};
  return error;
}

function actionableError(error) {
  const detail = error.details?.action;
  const analysis = error.details?.analysis;
  const excluded = analysis?.reason_counts?.length
    ? ` Exclusions: ${analysis.reason_counts.map((item) => `${item.reason_code} (${item.count})`).join(", ")}.`
    : "";
  return `Action stopped: ${error.message}.${detail ? ` ${detail}` : ""}${excluded}`;
}

function showRawEvidence(payload) {
  const text = JSON.stringify(payload, null, 2);
  rawEvidence.textContent = text.length > 24000
    ? `${text.slice(0, 24000)}\n… response collapsed at 24,000 characters; canonical evidence remains unchanged.`
    : text;
}

function consumeAction(action, result) {
  if (action === "resolve" && Array.isArray(result.candidates)) {
    renderResolvedCandidates(result.candidates);
  } else if (action === "compare") {
    latestComparison = result;
    blindAliases = [];
    renderComparison(result, false);
  } else if (action === "solo-review") {
    preferredReviewIdentity = result.review.identity.value;
  } else if (action === "blind-prepare") {
    blindAliases = result.packet.entries[0].outputs.map((item) => item.alias);
    renderBlindPacket(result.packet);
  } else if (action === "blind-seal") {
    preferredReviewIdentity = result.review.identity.value;
  } else if (action === "blind-reveal") {
    preferredReviewIdentity = result.review.identity.value;
    revealIdentities(result.candidate_mappings);
  } else if (action === "focused" || action === "batch" || action === "export") {
    renderLocalResult(result);
  } else if (action === "cleanup-preview" || action === "replay-plan") {
    rawEvidence.textContent = JSON.stringify(result, null, 2);
  } else if (action === "cleanup-execute" || action === "replay-execute") {
    rawEvidence.textContent = JSON.stringify(result, null, 2);
  }
  if (action === "launch") startOperationPolling();
}

function renderResolvedCandidates(candidates) {
  const target = document.querySelector("#candidate-resolution");
  target.replaceChildren(...candidates.map((candidate, index) => {
    const manifest = candidate.resolution.manifest;
    return element("section", "candidate-spec", [
      element("p", "eyebrow", `Candidate ${index + 1}`),
      element("h2", "", candidate.label),
      element(
        "span",
        "preflight-ready",
        candidate.preflight.status === "ready" ? "Preflight ready" : "Preflight blocked",
      ),
      definitionList({
        Rank: manifest.rank,
        Alpha: manifest.alpha,
        Seed: manifest.seed,
        Steps: manifest.training_steps,
        Targets: manifest.target_modules.join(", "),
        "System estimate": `${formatNumber(candidate.estimate.system_memory_bytes)} bytes`,
      }, "spec-list"),
    ]);
  }));
}

function renderWorkspace() {
  if (!workspace) return;
  renderMode();
  renderOverview();
  renderStages();
  renderDataset();
  renderResolutions();
  renderRuns();
  renderRecommendation();
  renderReviewCapture();
  renderStorage();
  renderInspector();
  renderOperation();
  syncCandidateDefaults();
}

function renderMode() {
  const real = workspace.mode === "real_local";
  const fixture = workspace.mode === "fixture_demo";
  const configured = real || fixture;
  const banner = document.querySelector("#mode-banner");
  banner.classList.toggle("is-demo", fixture);
  banner.classList.toggle("is-real", real);
  banner.classList.toggle("is-unconfigured", !configured);
  banner.replaceChildren(
    element("strong", "", real ? "REAL LOCAL TRAINING" : fixture ? "FIXTURE DEMO" : "CHOOSE A MODE"),
    element("span", "", workspace.mode_label || "Choose a mode to begin."),
  );
  document.querySelectorAll("[data-fixture-only]").forEach((node) => {
    node.classList.toggle("mode-hidden", real);
  });
  if (real && ["evaluate", "storage"].includes(activeStage)) showStage("use");
  if (configured) document.querySelector("#training-mode").value = workspace.mode;
  syncSetupControls();
  if (real && value("dataset-source-kind") === "fixture") {
    document.querySelector("#dataset-source-kind").value = "hugging_face";
  } else if (fixture && value("dataset-source-kind") === "hugging_face") {
    document.querySelector("#dataset-source-kind").value = "fixture";
  }
  document.querySelector("#runtime-label").textContent = real ? "Real local runtime" : fixture ? "Fixture demo runtime" : "Choose a mode";
  document.querySelector("#runtime-detail").textContent = real ? "No silent hardware fallback" : fixture ? "No model is trained" : "Not configured";
  const capability = workspace.capability;
  document.querySelector("#execution-context").textContent = real
    ? capability
      ? `${capability.accelerator_backend} / ${capability.accelerator_model}`
      : `${workspace.selected_target || "local target"} selected; preflight pending`
    : fixture ? "Fixture demo / offline" : "Not selected";
  document.querySelector("#run-kind-label").textContent = real ? "Real library runtime" : fixture ? "Fixture demo runtime" : "Runtime not selected";
  document.querySelector("#launch-action").textContent = real ? "Launch real training" : "Run demo";
  document.querySelector("#recipe-lead").textContent = real
    ? "One bounded LoRA recipe is preflighted against the selected model, tokenizer, dataset, backend, and hardware."
    : "Two deterministic fixture recipes exercise the workflow without training a model.";
  document.querySelector("#run-lead").textContent = real
    ? "Real progress, cancellation, terminal state, and verified adapter integrity stay together."
    : "Deterministic fixture payloads exercise progress and integrity handling; no model is trained.";
  document.querySelector("#use-lead").textContent = real
    ? "Use the single verified trained adapter for real local inference without creating a chat surface or deployment."
    : "Record the demo-output decision separately from evidence, then exercise deterministic local use without implying a trained model.";
  document.querySelector("#selection-copy").textContent = real
    ? "The single verified trained adapter is selected automatically"
    : "Choose the deterministic fixture output authorized for demo use";
  document.querySelector("#batch-lead").textContent = real
    ? "Run one local input per line against the selected trained adapter."
    : "Run one local input per line against the selected fixture payload.";
  document.querySelector("#export-kind").textContent = real ? "Portable adapter bundle" : "Fixture payload manifest";
  viewMeta.use[2] = real
    ? "Select one verified trained adapter for focused local inference and export."
    : "Select one verified fixture payload for deterministic demo use and export.";
  document.querySelector("#resolution-target-label").textContent = real
    ? capability
      ? `${capability.accelerator_backend} observed / no fallback used`
      : "Selected local target / preflight not yet observed"
    : "Portable fixture CPU / deterministic payloads";
  document.querySelector("#inspector-footnote").textContent = real
    ? "Canonical records remain immutable. Dataset import may use the selected public Hugging Face source; model execution remains local and uses no silent fallback."
    : fixture
      ? "Canonical records remain immutable. This fixture demo is offline and trains no model."
      : "Canonical records remain immutable. Choose a mode to begin.";
}

function renderOverview() {
  const stages = (workspace.stages || []).filter((stage) => stage.applicable !== false);
  const completed = stages.filter((stage) => stage.complete).length;
  const allComplete = stages.length > 0 && completed === stages.length;
  const started = Boolean(workspace.project);
  const next = stages.find((stage) => !stage.complete);
  const statusBoard = document.querySelector(".status-board");
  const badge = document.querySelector("#overview-status-badge");
  statusBoard?.classList.toggle("is-complete", allComplete);
  badge?.classList.toggle("is-complete", allComplete);
  if (badge) badge.textContent = allComplete ? "Journey verified" : started ? "In progress" : "Ready to begin";

  const journeyState = document.querySelector("#overview-journey-state");
  const journeyCopy = document.querySelector("#overview-journey-copy");
  if (journeyState) journeyState.textContent = allComplete ? "Verified" : started ? `${completed} / ${stages.length} stages complete` : "Not started";
  if (journeyCopy) journeyCopy.textContent = allComplete
    ? "Selection and local-use evidence are recorded."
    : next
      ? `Next: ${stageLabels[next.key]}.`
      : "Create the immutable project policy first.";

  const gate = document.querySelector("#overview-gate-state");
  if (gate) gate.textContent = workspace.store?.status === "verified" ? "Store verified" : "Awaiting records";
  document.querySelector("#overview-record-count").textContent = formatNumber(workspace.store?.record_count ?? 0);
  document.querySelector("#overview-artifact-count").textContent = formatNumber(workspace.artifacts?.length ?? 0);
  document.querySelector("#project-context").textContent = workspace.project?.display_name
    || (workspace.mode === "real_local"
      ? "Real local adapter project"
      : workspace.mode === "fixture_demo" ? "Fixture runtime project" : "Not configured");
  renderCandidateOverview();

  document.querySelectorAll("[data-journey-step]").forEach((item) => {
    const stage = stages.find((candidate) => candidate.key === item.dataset.journeyStep);
    item.classList.toggle("is-complete", Boolean(stage?.complete));
    item.classList.toggle("is-current", stage?.key === next?.key || (allComplete && stage?.key === "use"));
  });

  const action = document.querySelector("#overview-action");
  const title = document.querySelector("#next-action-title");
  const copy = document.querySelector("#next-action-copy");
  if (!action || !title || !copy) return;
  if (!started) {
    delete action.dataset.targetStage;
    action.textContent = "Create project";
    title.textContent = workspace.mode === "real_local" ? "Configure real local training" : "Choose and configure a mode";
    copy.textContent = workspace.mode === "real_local"
      ? "Confirm private local sources; the immutable project is created only after a successful dataset import."
      : "Choose the fixture demo or real local training contract.";
    return;
  }
  const destination = next?.key || "use";
  action.dataset.targetStage = destination;
  action.textContent = allComplete ? "Open local use" : `Open ${stageLabels[destination]}`;
  title.textContent = allComplete
    ? (workspace.mode === "real_local" ? "Continue with the selected adapter" : "Continue with the selected demo output")
    : `Continue to ${stageLabels[destination]}`;
  copy.textContent = allComplete
    ? (workspace.mode === "real_local"
      ? "The real local journey is complete and remains inspectable from the evidence ledger."
      : "The fixture demo journey is complete and remains inspectable from the evidence ledger.")
    : "The prior stage is recorded; continue when you are ready.";
}

function renderCandidateOverview() {
  const target = document.querySelector("#overview-candidates");
  if (!target) return;
  if (workspace.mode === "real_local") {
    const resolution = workspace.resolutions?.[0];
    const artifact = workspace.artifacts?.[0];
    target.replaceChildren(element("div", "candidate-table-row", [
      element("span", "candidate-primary", [element("strong", "", "Selected real candidate"), element("small", "", artifact?.label || "Verified adapter pending")]),
      element("span", "", resolution ? `Rank ${resolution.rank} / ${resolution.training_steps} steps` : "Preflight pending"),
      element("span", "", workspace.operation?.status || "idle"),
      element("span", artifact ? "candidate-good" : "", artifact?.integrity_status || "Pending"),
      element("span", artifact ? "candidate-good" : "", artifact ? "Selected by single-candidate default" : "Pending"),
    ]));
    return;
  }
  const resolutions = [...(workspace.resolutions || [])].sort((left, right) => left.rank - right.rank);
  const rows = [
    { key: "ember", label: "Ember / balanced", resolution: resolutions[0], runMatch: "run-fixture-runtime" },
    { key: "slate", label: "Slate / capacity", resolution: resolutions[1], runMatch: "run-fixture-challenger" },
  ];
  const currentDecision = (workspace.registry || []).find((item) => item.current);
  const hasCandidateState = rows.some((row) => row.resolution) || (workspace.artifacts || []).length > 0;
  if (!hasCandidateState) {
    target.replaceChildren(element("div", "empty-state", [
      element("strong", "", "No candidate evidence yet"),
      element("span", "", "Resolve recipes and run the fixture candidates to populate this ledger."),
    ]));
    return;
  }
  target.replaceChildren(
    element("div", "candidate-table-head", [
      element("span", "", "Candidate"),
      element("span", "", "Manifest"),
      element("span", "", "Run"),
      element("span", "", "Integrity"),
      element("span", "", "Registry"),
    ]),
    ...rows.map((row) => {
      const run = (workspace.runs || []).find((item) => item.run_id === row.runMatch);
      const artifact = (workspace.artifacts || []).find((item) => item.key === row.key);
      const selected = artifact && currentDecision
        && currentDecision.candidate.identity.value === artifact.reference.identity.value;
      return element("div", `candidate-table-row${artifact?.available ? " is-ready" : ""}`, [
        element("span", "candidate-primary", [
          element("strong", "", row.label),
          element("small", "", artifact?.label || "Fixture output pending"),
        ]),
        element("span", "", row.resolution ? `Rank ${row.resolution.rank} / seed ${row.resolution.seed}` : "Pending"),
        element("span", run?.status === "completed" ? "candidate-good" : "", run?.status || "Pending"),
        element("span", artifact?.integrity_status === "passed" ? "candidate-good" : "", artifact?.integrity_status || "Pending"),
        element("span", selected ? "candidate-good" : "", selected ? currentDecision.status : "Not selected"),
      ]);
    }),
  );
}

function renderReviewCapture() {
  const target = document.querySelector("#review-capture-selection");
  const reviews = completedReviews();
  const existing = target.value;
  target.replaceChildren(...(
    reviews.length
      ? reviews.map((review) => element(
        "option",
        "",
        `${review.mode} / ${review.reference.logical_id} / ${review.stage}`,
        { value: review.reference.identity.value },
      ))
      : [element("option", "", "No completed review recorded", { value: "" })]
  ));
  const requested = preferredReviewIdentity || existing;
  if (reviews.some((review) => review.reference.identity.value === requested)) {
    target.value = requested;
  }
  preferredReviewIdentity = null;
}

function completedReviews() {
  const completedStages = new Set(["recorded", "blind_sealed", "blind_revealed"]);
  return (workspace?.evaluation?.reviews || []).filter((review) => completedStages.has(review.stage));
}

function selectedCompletedReview() {
  const selected = value("review-capture-selection");
  const review = completedReviews().find((item) => item.reference.identity.value === selected);
  if (!review) throw new Error("completed_review_required");
  return review;
}

function renderStages() {
  (workspace.stages || []).forEach((stage) => {
    const buttons = document.querySelectorAll(`[data-stage="${stage.key}"]`);
    const badge = document.querySelector(`[data-stage-state="${stage.key}"]`);
    buttons.forEach((button) => button.classList.toggle("is-complete", stage.complete));
    badge?.classList.toggle("is-complete", stage.complete);
    if (badge) badge.textContent = stage.applicable === false ? "Not used" : stage.complete ? "Complete" : "Pending";
  });
}

function renderDataset() {
  const dataset = workspace.dataset;
  if (!dataset) {
    document.querySelectorAll("#dataset-ledger strong").forEach((node) => {
      node.textContent = "-";
    });
    document.querySelector("#dataset-previews").replaceChildren();
    document.querySelector("#dataset-analysis").replaceChildren();
    return;
  }
  const values = [
    dataset.statistics.accepted_rows,
    dataset.statistics.excluded_rows,
    dataset.statistics.total_tokens,
    dataset.rendered_bytes_count,
  ];
  document.querySelectorAll("#dataset-ledger strong").forEach((node, index) => {
    node.textContent = formatNumber(values[index]);
  });
  const analysis = dataset.analysis || {};
  const splitCounts = (analysis.split_counts || dataset.statistics.split_counts || [])
    .map((item) => `${item.split}: ${formatNumber(item.count)}`);
  const reasonCounts = (analysis.reason_counts || []).map((item) => `${item.reason_code}: ${formatNumber(item.count)}`);
  const sourceSummary = dataset.source
    ? `${dataset.source.kind || "local"} · ${formatNumber(dataset.source.imported_rows)} imported of ${formatNumber(dataset.source.available_rows ?? dataset.source.imported_rows)}`
    : workspace.mode === "fixture_demo"
      ? "Built-in fixture demo rows"
      : "Source metadata unavailable";
  document.querySelector("#dataset-analysis").replaceChildren(
    element("section", "analysis-block", [
      element("strong", "", "Split preflight"),
      element("span", "", splitCounts.join(" · ") || "No split counts available"),
    ]),
    element("section", "analysis-block", [
      element("strong", "", "Exclusion reasons"),
      element("span", "", reasonCounts.join(" · ") || "No rows excluded"),
    ]),
    element("section", "analysis-block", [
      element("strong", "", "Bounded source"),
      element("span", "", sourceSummary),
    ]),
  );
  const previews = document.querySelector("#dataset-previews");
  previews.replaceChildren(...(dataset.previews || []).map((preview) => element("details", "preview-row", [
    element("summary", "", `Row ${preview.source_ordinal} · ${preview.split} · ${formatNumber(preview.token_count)} tokens${preview.text_truncated ? " · preview truncated" : ""}`),
    element("pre", "preview-text", preview.text),
  ])));
  if (dataset.reimport_required) showNotice("Prepared bytes are not in memory after restart. Re-import before launching a pending run.", true);
}

function renderResolutions() {
  const target = document.querySelector("#candidate-resolution");
  if (!workspace.resolutions?.length) {
    target.replaceChildren(element("div", "empty-state", [
      element("strong", "", "No manifests yet"),
      element("span", "", "Resolve candidates to inspect settings and preflight evidence."),
    ]));
    return;
  }
  const byRank = [...workspace.resolutions].sort((left, right) => left.rank - right.rank);
  target.replaceChildren(...byRank.map((resolution, index) => {
    const key = workspace.mode === "real_local" ? "selected" : index === 0 ? "ember" : "slate";
    return element("section", "candidate-spec", [
      element("p", "eyebrow", `Candidate ${index + 1}`),
      element("h2", "", candidateLabel(key)),
      element("span", "preflight-ready", "Preflight ready"),
      definitionList({
        Rank: resolution.rank,
        Alpha: resolution.alpha,
        Seed: resolution.seed,
        Steps: resolution.training_steps,
        Targets: resolution.target_modules.join(", "),
        Precision: "fp32",
      }, "spec-list"),
    ]);
  }));
}

function renderRuns() {
  const target = document.querySelector("#run-timelines");
  if (workspace.mode === "real_local" && !workspace.runs?.length) {
    const operation = workspace.operation || {};
    target.replaceChildren(element("div", "empty-state", [
      element("strong", "", operation.status === "running" ? "Real training is running" : "No verified real artifact yet"),
      element("span", "", `${operation.phase || "Preflight and launch are required"}${operation.failure_code ? ` / ${operation.failure_code}` : ""}`),
    ]));
    return;
  }
  if (!workspace.runs?.length) return;
  target.replaceChildren(...workspace.runs.map((run) => {
    if (workspace.mode === "real_local") {
      const artifact = workspace.artifacts?.[0];
      return element("section", "run-timeline", [
        element("header", "", [element("strong", "", "Real local LoRA run"), element("span", "preflight-ready", run.status)]),
        element("p", "", artifact?.label || "No trained adapter artifact was verified."),
        element("p", artifact ? "artifact-line" : "artifact-line is-failed", artifact?.reference?.logical_id || "Artifact unavailable"),
      ]);
    }
    const candidateKey = run.run_id.includes("challenger") ? "slate" : "ember";
    const artifact = workspace.artifacts.find((item) => item.key === candidateKey);
    const progress = run.events.filter((event) => event.type === "run_progress");
    const last = progress.at(-1);
    const percent = last ? Math.round((last.step / last.total_steps) * 100) : 0;
    const logs = run.events.filter((event) => event.type === "run_log");
    return element("section", "run-timeline", [
      element("header", "", [
        element("strong", "", run.run_id.includes("challenger") ? "Slate / capacity" : "Ember / balanced"),
        element("span", "preflight-ready", run.status),
      ]),
      element(
        "div",
        "progress-track",
        element("span", percent >= 100 ? "is-full" : "is-partial", ""),
      ),
      element("p", "", last ? `Step ${last.step} / ${last.total_steps} / loss ${last.loss_microunits} micro-units` : "No progress evidence yet."),
      element("ul", "event-list", logs.map((log) => element("li", "", `${log.code} / ${log.step}`))),
      element(
        "p",
        `artifact-line${artifact && !artifact.available ? " is-failed" : ""}`,
        artifact
          ? `${artifact.label} / integrity ${artifact.integrity_status}${artifact.failure_code ? ` / ${artifact.failure_code}` : ""}`
          : "Artifact evidence pending.",
      ),
    ]);
  }));
}

function renderComparison(result, hidden) {
  const target = document.querySelector("#comparison-output");
  target.replaceChildren(...result.outputs.map((candidate) => element("section", "comparison-candidate", [
    element("header", "", [
      element("h2", "", candidate.label),
      element("span", `identity-alias${hidden ? " is-hidden" : ""}`, candidate.key, { "data-candidate-key": candidate.key }),
    ]),
    element("pre", "", JSON.stringify(candidate.output, null, 2)),
    element("span", "preflight-ready", "Integrity passed"),
  ])));
}

function renderBlindPacket(packet) {
  const target = document.querySelector("#comparison-output");
  const entry = packet.entries[0];
  target.replaceChildren(...entry.outputs.map((output) => element("section", "comparison-candidate", [
    element("header", "", [
      element("h2", "", output.alias),
      element("span", "identity-alias is-hidden", "identity sealed", { "data-blind-alias": output.alias }),
    ]),
    element("pre", "", JSON.stringify(output.output, null, 2)),
    element("span", "preflight-ready", "Leak audit passed"),
  ])));
}

function revealIdentities(mappings) {
  mappings.forEach((mapping) => {
    const node = document.querySelector(`[data-blind-alias="${mapping.alias}"]`);
    if (!node) return;
    node.textContent = mapping.candidate.logical_id.includes("challenger") ? "Slate / capacity" : "Ember / balanced";
    node.classList.remove("is-hidden");
  });
}

function renderRecommendation() {
  const target = document.querySelector("#recommendation-view");
  const recommendation = workspace.recommendation;
  if (!recommendation) {
    target.replaceChildren();
    return;
  }
  const selected = recommendation.selected_candidate;
  if (workspace.mode === "real_local") {
    target.replaceChildren(element("div", "recommendation-banner", [
      element("div", "", [element("span", "", "Single-candidate default"), element("strong", "", selected ? "Verified real adapter" : "Awaiting verified training")]),
      element("span", "", "No fixture candidate is substituted"),
    ]));
    return;
  }
  const label = selected ? (selected.logical_id.includes("challenger") ? "Slate / capacity" : "Ember / balanced") : "No qualified candidate";
  const conflicts = recommendation.conflicts || [];
  target.replaceChildren(element("div", "recommendation-banner", [
    element("div", "", [
      element("p", "eyebrow", `Confidence / ${recommendation.confidence}`),
      element("strong", "", label),
      element("p", "conflict-line", conflicts.join(" / ") || "No disclosed conflict"),
    ]),
    element("span", "stage-state is-complete", "Policy derived"),
  ]));
}

function renderLocalResult(result) {
  const target = document.querySelector("#local-output");
  const text = JSON.stringify(result, null, 2);
  target.replaceChildren(
    element("p", "eyebrow", result.status === "verified" ? "Verified export" : "Local runtime evidence"),
    element("pre", "bounded-output", text.length > 6000 ? `${text.slice(0, 6000)}\n… output collapsed; inspect the bounded evidence response for more.` : text),
  );
}

function renderStorage() {
  const retention = workspace.retention || {};
  document.querySelector("#retention-default").textContent = capitalize(retention.retention_default || "full");
  document.querySelector("#storage-logical-bytes").textContent = formatBytes(retention.logical_bytes);
  document.querySelector("#storage-physical-bytes").textContent = formatBytes(retention.physical_bytes);
  document.querySelector("#storage-reclaimable-bytes").textContent = formatBytes(retention.reclaimable_physical_bytes);
  document.querySelector("#storage-entry-count").textContent = `${formatNumber(retention.entry_count)} objects`;

  const selected = new Set(retention.active_plan?.selected_entry_ids || []);
  const target = document.querySelector("#storage-entries");
  const entries = retention.entries || [];
  target.replaceChildren(...(
    entries.length
      ? entries.map((entry) => {
        const attributes = {
          type: "checkbox",
          value: entry.entry_id,
          "aria-label": `Select ${entry.logical_key}`,
        };
        attributes["data-deletable"] = String(entry.deletable);
        if (!entry.deletable || retention.active_plan) attributes.disabled = "";
        if (selected.has(entry.entry_id)) attributes.checked = "";
        const checkbox = element("input", "storage-entry-select", [], attributes);
        checkbox.addEventListener("change", handleStorageSelectionChange);
        return element("tr", entry.deletable ? "" : "is-protected", [
          element("td", "storage-select", checkbox),
          element("td", "storage-object", [
            element("strong", "", entry.logical_key),
            element("small", "", entry.deletable ? entry.entry_id : "Protected runtime evidence"),
          ]),
          element("td", "", entry.byte_class.replaceAll("_", " ")),
          element("td", "storage-size", formatBytes(entry.byte_count)),
          element("td", "", `${entry.local_reference_count} local / ${entry.external_reference_count} external`),
        ]);
      })
      : [element("tr", "", element("td", "storage-empty", "No runtime bytes have been created yet.", { colspan: "5" }))]
  ));
  updateStorageSelectionSummary();
  renderCleanupPlan(retention.active_plan, retention.receipts || []);
  renderReplayPlan(workspace.reproduction || {});
  syncStorageControlState();
}

function handleStorageSelectionChange() {
  if (workspace?.retention?.active_plan) {
    workspace.retention.active_plan = null;
    renderCleanupPlan(null, workspace.retention.receipts || []);
  }
  updateStorageSelectionSummary();
  syncStorageControlState();
}

function updateStorageSelectionSummary() {
  const ids = new Set(selectedStorageEntryIds());
  const entries = workspace?.retention?.entries || [];
  const bytes = entries
    .filter((entry) => ids.has(entry.entry_id))
    .reduce((total, entry) => total + entry.byte_count, 0);
  const target = document.querySelector("#cleanup-selection-summary");
  if (!target) return;
  target.textContent = ids.size
    ? `${ids.size} explicit object${ids.size === 1 ? "" : "s"} / ${formatBytes(bytes)} logical bytes selected.`
    : "Select explicit deletable objects to calculate consequences.";
}

function selectedStorageEntryIds() {
  return [...document.querySelectorAll(".storage-entry-select:checked")].map((item) => item.value);
}

function renderCleanupPlan(plan, receipts) {
  const title = document.querySelector("#cleanup-plan-title");
  const target = document.querySelector("#cleanup-plan-view");
  const confirmation = document.querySelector("#cleanup-confirmation");
  const planKey = plan ? `${plan.plan_id}:${plan.execution_id || ""}` : null;
  if (planKey !== renderedCleanupPlanKey) confirmation.checked = false;
  renderedCleanupPlanKey = planKey;
  if (plan) {
    title.textContent = `${formatBytes(plan.physical_bytes_freed)} physically reclaimable`;
    const warnings = plan.warnings || [];
    target.replaceChildren(
      element("div", "consequence-metrics", [
        element("div", "", [element("span", "", "Logical removal"), element("strong", "", formatBytes(plan.logical_bytes_selected))]),
        element("div", "", [element("span", "", "Physical freed"), element("strong", "", formatBytes(plan.physical_bytes_freed))]),
      ]),
      element("p", "consequence-label", "Impact warnings"),
      ...(warnings.length
        ? warnings.map((warning) => element("div", "impact-warning", [
          element("strong", "", warning.category.replaceAll("_", " ")),
          element("span", "", `${warning.entry_ids.length} affected object${warning.entry_ids.length === 1 ? "" : "s"}`),
        ]))
        : [element("p", "", "No capability loss was identified for this exact selection.")]),
      element("p", "retained-classes", `Retained classes: ${(plan.retained_byte_classes || []).join(", ") || "none"}`),
    );
    return;
  }
  const latest = receipts.at(-1);
  if (latest) {
    title.textContent = `Recorded cleanup ${latest.outcome}`;
    target.replaceChildren(
      element("div", `cleanup-result cleanup-${latest.outcome}`, [
        element("strong", "", `${formatBytes(latest.physical_bytes_freed)} physically freed`),
        element("span", "", `${formatBytes(latest.logical_bytes_removed)} logical bytes removed`),
        element("small", "", latest.failure_code || "Immutable receipt recorded"),
      ]),
      element("p", "retained-classes", "Canonical manifests and lifecycle evidence remain in the project store."),
    );
    return;
  }
  title.textContent = "Nothing is selected.";
  target.replaceChildren(element("p", "", "Temper will calculate physical bytes freed, retained classes, affected records, and every loss of capability before deletion."));
}

function renderReplayPlan(reproduction) {
  const target = document.querySelector("#replay-plan-view");
  const plan = reproduction.active_plan;
  if (plan) {
    const exact = plan.mode === "strict_replay";
    target.replaceChildren(
      element("p", "card-kicker", exact ? "Exact reproduction" : "Adapted reproduction"),
      element("h3", "", exact ? "Manifest identity preserved" : "New derived experiment"),
      element("div", "manifest-identity-pair", [
        element("div", "", [element("span", "", "Source"), element("code", "", shortIdentity(plan.source_manifest_identity?.value))]),
        element("span", "manifest-arrow", exact ? "=" : "→"),
        element("div", "", [element("span", "", "Planned"), element("code", "", shortIdentity(plan.planned_manifest_identity?.value))]),
      ]),
      ...(plan.manifest_changes || []).map((change) => element("div", "manifest-change", [
        element("code", "", change.path),
        element("span", "", change.operation),
        element("small", "", `${manifestValue(change.before)} → ${manifestValue(change.after)}`),
      ])),
      element("p", `replay-status replay-${plan.status}`, plan.status === "ready" ? "Ready to execute as a new run." : `Blocked: ${(plan.reasons || []).join(", ")}`),
    );
    return;
  }
  const latestExecution = (reproduction.executions || []).at(-1);
  const latestDerivation = (reproduction.derivations || []).at(-1);
  if (latestExecution || latestDerivation) {
    target.replaceChildren(
      element("p", "card-kicker", latestExecution?.adapted_reproduction ? "Adapted reproduction" : "Strict replay"),
      element("h3", "", latestExecution ? `Replay ${latestExecution.status}` : "Derived experiment recorded"),
      element("p", "", latestExecution?.run_id || latestDerivation?.reason || "Reproduction evidence is available."),
      ...((latestDerivation?.manifest_changes || []).map((change) => element("div", "manifest-change", [
        element("code", "", change.path),
        element("span", "", change.operation),
        element("small", "", `${manifestValue(change.before)} → ${manifestValue(change.after)}`),
      ]))),
    );
    return;
  }
  target.replaceChildren(
    element("p", "card-kicker", "Manifest comparison"),
    element("h3", "", "No replay planned"),
    element("p", "", "Choose a candidate and mode to compare the source and planned manifests before launch."),
  );
}

function renderInspector() {
  const project = workspace.project;
  const projectName = project?.display_name
    || (workspace.mode === "real_local"
      ? "Real local adapter project"
      : workspace.mode === "fixture_demo" ? "Fixture runtime project" : "Not configured");
  document.querySelector("#inspector-status").textContent = project
    ? `${projectName} / ${workspace.store?.status || "active"}`
    : "Awaiting project setup.";
  const ledger = {
    Records: workspace.store?.record_count ?? 0,
    Events: workspace.store?.event_count ?? 0,
    Artifacts: workspace.artifacts?.length ?? 0,
    Reviews: workspace.evaluation?.reviews?.length ?? 0,
    Decisions: workspace.registry?.length ?? 0,
    Sessions: workspace.local_use?.saved_session_count ?? 0,
    Exports: workspace.local_use?.export_count ?? 0,
    "Physical bytes": formatBytes(workspace.retention?.physical_bytes),
    "Cleanup receipts": workspace.retention?.receipts?.length ?? 0,
    Replays: workspace.reproduction?.executions?.length ?? 0,
    Deployment: "none",
  };
  document.querySelector("#inspector-ledger").replaceChildren(...Object.entries(ledger).flatMap(([key, item]) => [
    element("dt", "", key),
    element("dd", "", String(item)),
  ]));
  showRawEvidence(workspace);
}

function renderOperation() {
  const rail = document.querySelector("#operation-rail");
  const summary = document.querySelector("#operation-summary");
  const operation = workspace.operation || { status: "idle", phase: "not_started", elapsed_seconds: 0 };
  const progress = operation.total_steps
    ? `Step ${operation.step} / ${operation.total_steps}`
    : "No step evidence yet";
  rail.classList.toggle("is-running", operation.status === "running");
  rail.classList.toggle("is-failed", operation.status === "failed");
  rail.replaceChildren(
    element("span", "operation-dot", "", { "aria-hidden": "true" }),
    element("strong", "", capitalize(operation.phase.replaceAll("_", " "))),
    element("span", "", `${progress} · ${operation.elapsed_seconds || 0}s${operation.failure_code ? ` · ${operation.failure_code}` : ""}`),
  );
  summary.textContent = operation.status === "running"
    ? `${capitalize(operation.phase)} · ${progress}`
    : operation.recovery_action || `${capitalize(operation.status)} · ${capitalize(operation.phase)}`;
  const cancel = document.querySelector('[data-action="cancel"]');
  if (cancel) cancel.disabled = operation.status !== "running";
  const launch = document.querySelector('[data-action="launch"]');
  if (launch) {
    launch.disabled = operation.status === "running"
      || (workspace.mode === "real_local" && !workspace.resolutions?.length);
  }
  const realUseBlocked = workspace.mode === "real_local" && !realAdapterReady();
  for (const action of ["focused", "batch", "export"]) {
    const control = document.querySelector(`[data-action="${action}"]`);
    if (control) control.disabled = realUseBlocked;
  }
  if (operation.status === "running") startOperationPolling();
  else stopOperationPolling();
}

function startOperationPolling() {
  if (operationPoll) return;
  operationPoll = window.setInterval(loadWorkspace, 1000);
}

function stopOperationPolling() {
  if (!operationPoll) return;
  window.clearInterval(operationPoll);
  operationPoll = null;
}

function syncCandidateDefaults() {
  const real = workspace.mode === "real_local";
  const ready = !real || realAdapterReady();
  const preferred = real
    ? ready ? "selected" : ""
    : workspace.recommendation?.selected_candidate?.logical_id?.includes("challenger") ? "slate" : "ember";
  const choices = real
    ? ready
      ? [{ value: "selected", label: "Verified real adapter" }]
      : [{ value: "", label: "No verified real adapter yet" }]
    : [
      { value: "ember", label: "Ember / balanced" },
      { value: "slate", label: "Slate / capacity" },
    ];
  for (const id of ["candidate-selection", "replay-candidate"]) {
    const select = document.getElementById(id);
    if (!select) continue;
    select.replaceChildren(...choices.map((choice) => element("option", "", choice.label, { value: choice.value })));
    select.value = preferred;
  }
}

function realAdapterReady() {
  return (workspace?.artifacts || []).some((artifact) => (
    artifact.artifact_kind === "real_trained_lora_adapter"
      && artifact.integrity_status === "verified"
      && artifact.available === true
  ));
}

function showStage(stage) {
  if (workspace?.mode === "real_local" && ["evaluate", "storage"].includes(stage)) stage = "use";
  activeStage = stage;
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    const active = panel.dataset.panel === stage;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });
  document.querySelectorAll("[data-stage]").forEach((button) => {
    if (button.dataset.stage === stage) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  const meta = viewMeta[stage];
  if (meta) {
    document.querySelector("#view-kicker").textContent = meta[0];
    document.querySelector("#view-title").textContent = meta[1];
    document.querySelector("#view-subtitle").textContent = meta[2];
  }
  const panelTitle = document.querySelector(`[data-panel="${activeStage}"] [data-panel-title]`);
  if (panelTitle instanceof HTMLElement) panelTitle.focus({ preventScroll: true });
}

function showNotice(message, isError) {
  notice.hidden = false;
  notice.textContent = message;
  notice.classList.toggle("is-error", isError);
}

function setBusy(busy, action = null) {
  if (busy && !["launch", "cancel"].includes(action)) {
    clientOperationStartedAt = performance.now();
    clientOperationAction = action;
    renderClientOperation();
    if (clientOperationTimer) window.clearInterval(clientOperationTimer);
    clientOperationTimer = window.setInterval(() => renderClientOperation(), 1000);
  } else if (!busy) {
    if (clientOperationTimer) window.clearInterval(clientOperationTimer);
    clientOperationTimer = null;
    clientOperationStartedAt = null;
    clientOperationAction = null;
  }
  document.querySelectorAll("button, input, select, textarea").forEach((control) => {
    if (busy) {
      if (control.dataset.disabledBeforeBusy === undefined) {
        control.dataset.disabledBeforeBusy = String(control.disabled);
      }
      control.disabled = true;
    } else if (control.dataset.disabledBeforeBusy !== undefined) {
      control.disabled = control.dataset.disabledBeforeBusy === "true";
      delete control.dataset.disabledBeforeBusy;
    }
  });
  document.body.setAttribute("aria-busy", String(busy));
  if (!busy) syncStorageControlState();
}

function renderClientOperation(detail = null) {
  if (clientOperationStartedAt === null) return;
  const labels = {
    setup: "Saving mode and local source configuration",
    import: "Importing, tokenizing, and analyzing bounded data",
    resolve: "Loading sources and probing backend compatibility",
    compare: "Running synchronized fixture comparison",
    evaluate: "Evaluating fixture evidence",
    focused: "Running verified local inference",
    batch: "Running verified local batch",
    export: "Writing and re-verifying portable export",
    "replay-plan": "Planning reproduction",
    "replay-execute": "Executing reproduction",
  };
  const label = detail || labels[clientOperationAction] || "Working";
  const elapsed = Math.max(0, Math.floor((performance.now() - clientOperationStartedAt) / 1000));
  const rail = document.querySelector("#operation-rail");
  rail.classList.add("is-running");
  rail.replaceChildren(
    element("span", "operation-dot", "", { "aria-hidden": "true" }),
    element("strong", "", label),
    element("span", "", `${elapsed}s elapsed · cancellation is available during real training`),
  );
}

function syncStorageControlState() {
  const cleanupPlan = workspace?.retention?.active_plan;
  document.querySelectorAll(".storage-entry-select").forEach((checkbox) => {
    checkbox.disabled = Boolean(cleanupPlan) || checkbox.dataset.deletable !== "true";
  });
  const preview = document.querySelector('[data-action="cleanup-preview"]');
  const execute = document.querySelector('[data-action="cleanup-execute"]');
  const confirmation = document.querySelector("#cleanup-confirmation");
  if (preview) preview.disabled = Boolean(cleanupPlan) || !selectedStorageEntryIds().length;
  if (execute) execute.disabled = !cleanupPlan;
  if (confirmation) confirmation.disabled = !cleanupPlan;

  const replayPlan = workspace?.reproduction?.active_plan;
  const replayCandidate = document.querySelector("#replay-candidate");
  const replayMode = document.querySelector("#replay-mode");
  const replayPreview = document.querySelector('[data-action="replay-plan"]');
  const replayExecute = document.querySelector('[data-action="replay-execute"]');
  if (replayCandidate) replayCandidate.disabled = Boolean(replayPlan);
  if (replayMode) replayMode.disabled = Boolean(replayPlan);
  if (replayPreview) replayPreview.disabled = Boolean(replayPlan);
  if (replayExecute) replayExecute.disabled = !replayPlan || replayPlan.status !== "ready";
}

function sameStrings(left, right) {
  return left.length === right.length && left.every((item, index) => item === right[index]);
}

function reviewBody() {
  return {
    notes: value("review-notes"),
    ratings: { ember: numberValue("rating-ember"), slate: numberValue("rating-slate") },
    declaration: "I reviewed the synchronized prompt, settings, and both recorded outputs.",
  };
}

function blindReviewBody() {
  const aliases = blindAliases.length ? blindAliases : ["candidate-001", "candidate-002"];
  return {
    notes: value("review-notes"),
    ratings: Object.fromEntries(aliases.map((alias, index) => [alias, index === 0 ? numberValue("rating-ember") : numberValue("rating-slate")])),
    declaration: "I recorded this judgment before candidate identities were revealed.",
  };
}

function successMessage(action) {
  return {
    setup: workspace?.mode === "real_local"
      ? "Real local configuration saved. The immutable project is created only after a successful dataset import."
      : "Fixture demo project opened with immutable policy evidence.",
    import: "Dataset version frozen; previews remain local to this session.",
    resolve: workspace?.mode === "real_local"
      ? "The real LoRA preflight recorded source, backend, target-module, and hardware status; no fallback was used."
      : "Both demo recipes resolved and passed fixture CPU preflight.",
    launch: workspace?.mode === "real_local"
      ? "Real training started; progress will refresh until a terminal state."
      : "Both demo runs completed with verified fixture payloads; no model was trained.",
    compare: "Synchronized comparison completed.",
    "solo-review": "Structured solo review recorded.",
    "blind-prepare": "Blind packet prepared and leak-audited.",
    "blind-seal": "Judgment sealed before reveal.",
    "blind-reveal": "Candidate mapping revealed from sealed evidence.",
    evaluate: "Policy recommendation recorded with tie disclosure.",
    capture: "Selected completed review converted into a development case.",
    select: "Fixture user selection recorded separately from the recommendation.",
    focused: "Focused local-use session saved without chat memory.",
    batch: "Local batch completed with shared settings.",
    export: workspace?.mode === "real_local"
      ? "Portable adapter export verified; no deployment was created."
      : "Fixture payload manifest verified; no trained adapter or deployment was created.",
    "cleanup-preview": "Exact cleanup consequences calculated; no bytes were removed.",
    "cleanup-execute": "Cleanup finished and an immutable receipt was recorded.",
    "replay-plan": "Replay plan calculated with exact manifest identities.",
    "replay-execute": "Reproduction ended with its recorded immutable run outcome.",
  }[action] || "Action completed.";
}

function candidateLabel(key) {
  if (key === "selected") return "Selected real candidate";
  return key === "ember" ? "Ember / balanced" : "Slate / capacity";
}

function value(id) {
  return document.getElementById(id).value;
}

function numberValue(id) {
  return Number.parseInt(value(id), 10);
}

function optionalNumberValue(id) {
  const text = value(id).trim();
  return text ? Number.parseInt(text, 10) : undefined;
}

function formatNumber(item) {
  return Number(item || 0).toLocaleString("en-US");
}

function formatBytes(item) {
  const bytes = Number(item || 0);
  if (bytes < 1024) return `${formatNumber(bytes)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = "B";
  for (const candidate of units) {
    value /= 1024;
    unit = candidate;
    if (value < 1024) break;
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${unit}`;
}

function capitalize(item) {
  const text = String(item || "");
  return text ? `${text[0].toUpperCase()}${text.slice(1)}` : "";
}

function shortIdentity(value) {
  return typeof value === "string" ? `${value.slice(0, 12)}…` : "pending";
}

function manifestValue(value) {
  if (value === undefined) return "∅";
  const encoded = JSON.stringify(value);
  return encoded.length > 52 ? `${encoded.slice(0, 49)}…` : encoded;
}

function definitionList(values, className) {
  return element("dl", className, Object.entries(values).flatMap(([key, item]) => [
    element("div", "", [element("dt", "", key), element("dd", "", String(item))]),
  ]));
}

function element(tag, className = "", children = [], attributes = {}) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  const values = Array.isArray(children) ? children : [children];
  values.forEach((child) => {
    if (child instanceof Node) node.append(child);
    else if (child !== undefined && child !== null) node.append(document.createTextNode(String(child)));
  });
  Object.entries(attributes).forEach(([key, item]) => node.setAttribute(key, item));
  return node;
}
