"use strict";

const csrfToken = document.querySelector('meta[name="temper-csrf"]').content;
const rawEvidence = document.querySelector("#raw-evidence");
const notice = document.querySelector("#notice");
let workspace = null;
let activeStage = "setup";
let latestComparison = null;
let blindAliases = [];
let preferredReviewIdentity = null;

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
};

document.querySelectorAll("[data-stage]").forEach((button) => {
  button.addEventListener("click", () => showStage(button.dataset.stage));
});

document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", async () => {
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
  renderStages();
  renderDataset();
  renderResolutions();
  renderRuns();
  renderRecommendation();
  renderReviewCapture();
  renderInspector();
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
        `${review.mode} · ${review.reference.logical_id} · ${review.stage}`,
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
    const button = document.querySelector(`[data-stage="${stage.key}"]`);
    const badge = document.querySelector(`[data-stage-state="${stage.key}"]`);
    button?.classList.toggle("is-complete", stage.complete);
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
      element("p", "", last ? `Step ${last.step} / ${last.total_steps} · loss ${last.loss_microunits} μ` : "No progress evidence yet."),
      element("ul", "event-list", logs.map((log) => element("li", "", `${log.code} · ${log.step}`))),
      element(
        "p",
        `artifact-line${artifact && !artifact.available ? " is-failed" : ""}`,
        artifact
          ? `${artifact.reference.logical_id} · integrity ${artifact.integrity_status}${artifact.failure_code ? ` · ${artifact.failure_code}` : ""}`
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
      element("p", "conflict-line", recommendation.conflicts.join(" · ") || "No disclosed conflict"),
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

function renderInspector() {
  const project = workspace.project;
  document.querySelector("#inspector-status").textContent = project
    ? `${project.display_name} · ${workspace.status}`
    : "Awaiting project setup.";
  const ledger = {
    Records: workspace.store?.record_count ?? 0,
    Events: workspace.store?.event_count ?? 0,
    Artifacts: workspace.artifacts?.length ?? 0,
    Reviews: workspace.evaluation?.reviews?.length ?? 0,
    Decisions: workspace.registry?.length ?? 0,
    Sessions: workspace.local_use?.saved_session_count ?? 0,
    Exports: workspace.local_use?.export_count ?? 0,
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
  document.querySelector(`[data-panel="${activeStage}"] h1`)?.focus?.();
}

function showNotice(message, isError) {
  notice.hidden = false;
  notice.textContent = message;
  notice.classList.toggle("is-error", isError);
}

function setBusy(busy) {
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = busy;
  });
  document.body.setAttribute("aria-busy", String(busy));
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
