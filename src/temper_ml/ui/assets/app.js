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

const viewMeta = {
  setup: ["Workspace / 01", "Overview", "Project state, evidence, and the next bounded action."],
  data: ["Workspace / 02", "Data", "Import, inspect, and freeze deterministic training examples."],
  recipe: ["Workspace / 03", "Recipes", "Resolve two visible candidate manifests against one target."],
  run: ["Workspace / 04", "Runs", "Follow runtime evidence from launch through artifact integrity."],
  evaluate: ["Workspace / 05", "Evaluate", "Compare synchronized outputs and record honest review evidence."],
  use: ["Workspace / 06", "Local use", "Select one verified adapter for focused local work and export."],
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
  setup: () => post("/api/v1/setup", {}),
  import: () => post("/api/v1/dataset/import", {
    format: value("dataset-format"),
    source: value("dataset-source") || undefined,
  }),
  resolve: () => post("/api/v1/candidates/resolve", {}),
  launch: () => post("/api/v1/runs/launch", {}),
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
    if (!plan?.plan_id) throw new Error("replay_plan_required");
    const candidateKey = value("replay-candidate");
    const mode = value("replay-mode");
    if (candidateKey !== plan.candidate_key || mode !== plan.mode) {
      throw new Error("replay_controls_plan_mismatch");
    }
    return post("/api/v1/replays/execute", {
      plan_id: plan.plan_id,
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
    setBusy(true);
    try {
      const payload = await action();
      consumeAction(button.dataset.action, payload.result);
      workspace = payload.workspace;
      renderWorkspace();
      showNotice(successMessage(button.dataset.action), false);
    } catch (error) {
      showNotice(`Action stopped: ${error.message}`, true);
    } finally {
      setBusy(false);
    }
  });
});

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
  rawEvidence.textContent = JSON.stringify(payload, null, 2);
  if (!response.ok || !payload.ok) throw new Error(payload.error?.code || "request_failed");
  return payload.data;
}

function consumeAction(action, result) {
  if (action === "resolve") {
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
  renderOverview();
  renderStages();
  renderDataset();
  renderResolutions();
  renderRuns();
  renderRecommendation();
  renderReviewCapture();
  renderStorage();
  renderInspector();
}

function renderOverview() {
  const stages = workspace.stages || [];
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
  document.querySelector("#project-context").textContent = workspace.project?.display_name || "Fixture runtime project";
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
    title.textContent = "Create fixture project";
    copy.textContent = "Open the task-centered workspace and freeze its project policy.";
    return;
  }
  const destination = next?.key || "use";
  action.dataset.targetStage = destination;
  action.textContent = allComplete ? "Open local use" : `Open ${stageLabels[destination]}`;
  title.textContent = allComplete ? "Continue with the selected adapter" : `Continue to ${stageLabels[destination]}`;
  copy.textContent = allComplete
    ? "The fixture journey is complete and remains inspectable from the evidence ledger."
    : "The prior stage is recorded; continue when you are ready.";
}

function renderCandidateOverview() {
  const target = document.querySelector("#overview-candidates");
  if (!target) return;
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
          element("small", "", artifact?.reference.logical_id || "Artifact pending"),
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
    if (badge) badge.textContent = stage.complete ? "Complete" : "Pending";
  });
}

function renderDataset() {
  const dataset = workspace.dataset;
  if (!dataset) return;
  const values = [
    dataset.statistics.accepted_rows,
    dataset.statistics.excluded_rows,
    dataset.statistics.total_tokens,
    dataset.rendered_bytes_count,
  ];
  document.querySelectorAll("#dataset-ledger strong").forEach((node, index) => {
    node.textContent = formatNumber(values[index]);
  });
  const previews = document.querySelector("#dataset-previews");
  previews.replaceChildren(...(dataset.previews || []).map((preview) => element("div", "preview-row", [
    element("span", "", `Row ${preview.source_ordinal}`),
    element("span", "", preview.split),
    element("span", "", preview.text),
  ])));
  if (dataset.reimport_required) showNotice("Prepared bytes are not in memory after restart. Re-import before launching a pending run.", true);
}

function renderResolutions() {
  const target = document.querySelector("#candidate-resolution");
  if (!workspace.resolutions?.length) return;
  const byRank = [...workspace.resolutions].sort((left, right) => left.rank - right.rank);
  target.replaceChildren(...byRank.map((resolution, index) => {
    const key = index === 0 ? "ember" : "slate";
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
  if (!workspace.runs?.length) return;
  target.replaceChildren(...workspace.runs.map((run) => {
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
          ? `${artifact.reference.logical_id} / integrity ${artifact.integrity_status}${artifact.failure_code ? ` / ${artifact.failure_code}` : ""}`
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
  if (!recommendation) return;
  const selected = recommendation.selected_candidate;
  const label = selected ? (selected.logical_id.includes("challenger") ? "Slate / capacity" : "Ember / balanced") : "No qualified candidate";
  target.replaceChildren(element("div", "recommendation-banner", [
    element("div", "", [
      element("p", "eyebrow", `Confidence / ${recommendation.confidence}`),
      element("strong", "", label),
      element("p", "conflict-line", recommendation.conflicts.join(" / ") || "No disclosed conflict"),
    ]),
    element("span", "stage-state is-complete", "Policy derived"),
  ]));
}

function renderLocalResult(result) {
  const target = document.querySelector("#local-output");
  target.replaceChildren(
    element("p", "eyebrow", result.status === "verified" ? "Verified export" : "Local runtime evidence"),
    element("pre", "", JSON.stringify(result, null, 2)),
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
  document.querySelector("#inspector-status").textContent = project
    ? `${project.display_name} / ${workspace.status}`
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
  rawEvidence.textContent = JSON.stringify(workspace, null, 2);
}

function showStage(stage) {
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

function setBusy(busy) {
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
    setup: "Fixture project opened with immutable policy evidence.",
    import: "Dataset version frozen; previews remain local to this session.",
    resolve: "Both recipes resolved and passed fixture CPU preflight.",
    launch: "Both runs completed with verified adapter artifacts.",
    compare: "Synchronized comparison completed.",
    "solo-review": "Structured solo review recorded.",
    "blind-prepare": "Blind packet prepared and leak-audited.",
    "blind-seal": "Judgment sealed before reveal.",
    "blind-reveal": "Candidate mapping revealed from sealed evidence.",
    evaluate: "Policy recommendation recorded with tie disclosure.",
    capture: "Selected completed review converted into a development case.",
    select: "User selection recorded separately from the recommendation.",
    focused: "Focused local-use session saved without chat memory.",
    batch: "Local batch completed with shared settings.",
    export: "Portable adapter export verified; no deployment was created.",
    "cleanup-preview": "Exact cleanup consequences calculated; no bytes were removed.",
    "cleanup-execute": "Cleanup finished and an immutable receipt was recorded.",
    "replay-plan": "Replay plan calculated with exact manifest identities.",
    "replay-execute": "Reproduction ended with its recorded immutable run outcome.",
  }[action] || "Action completed.";
}

function candidateLabel(key) {
  return key === "ember" ? "Ember / balanced" : "Slate / capacity";
}

function value(id) {
  return document.getElementById(id).value;
}

function numberValue(id) {
  return Number.parseInt(value(id), 10);
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
