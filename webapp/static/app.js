const form = document.getElementById("predict-form");
const dnaInput = document.getElementById("dna-sequence");
const proteinInput = document.getElementById("protein-sequence");
const predictButton = document.getElementById("predict-button");
const loadingIndicator = document.getElementById("loading-indicator");
const errorBox = document.getElementById("error-box");
const infoBox = document.getElementById("info-box");
const emptyState = document.getElementById("empty-state");
const resultsContent = document.getElementById("results-content");
const probabilityValue = document.getElementById("probability-value");
const classValue = document.getElementById("class-value");
const logitValue = document.getElementById("logit-value");
const inputSummary = document.getElementById("input-summary");
const normalizedSummary = document.getElementById("normalized-summary");
const peakSummary = document.getElementById("peak-summary");
const dnaSequenceView = document.getElementById("dna-sequence-view");
const proteinSequenceView = document.getElementById("protein-sequence-view");
const exampleButtons = document.querySelectorAll(".example-button");

const EXAMPLES = {
  researcher: {
    protein:
      "MEGDAVEAIVEESETFIKGKERKTYQRRREGGQEEDACHLPQNQTDGGEVVQDVNSSVQMVMMEQLDPTLLQMKTEVMEGTVAPEAEAAVDDTQIITLQVVNMEEQPINIGELQLVQVPVPVTVPVATTSVEELQGAYENEVSKEGLAESEPMICHTLPLPEGFQVVKVGANGEVETLEQGELPPQEDPSWQKDPDYQPPAKKTKKTKKSKLRYTEEGKDVDVSVYDFEEEQQEGLLSEVNAEKVVGNMKPPKPTKIKKKGVKKTFQCELCSYTCPRRSNLDRHMKSHTDERPHKCHLCGRAFRTVTLLRNHLNTHTGTRPHKCPDCDMAFVTSGELVRHRRYKHTHEKPFKCSMCDYASVEVSKLKRHIRSHTGERPFQCSLCSYASRDTYKLKRHMRTHSGEKPYECYICHARFTQSGTMKMHILQKHTENVAKFHCPHCDTVIARKSDLGVHLRKQHSYIEQGKKCRYCDAVFHERYALIQHQKSHKNEKRFKCDQCDYACRQERHMIMHKRTHTGEKPYACSHCDKTFRQKQLLDMHFKRYHDPNFVPAAFVCSKCGKTFTRRNTMARHADNCAGPDGVEGENGGETKKSKRGRKRKMRSKKEDSSDSENAEPDLDDNEDEEEPAVEIEPEPEPQPVTPAPPPAKKRRGRPPGRTNQPKQNQPTAIIQVEDQNTGAIENIIVEVKKEPDAEPAEGEEEEAQPAATDAPNGDLTPEMILSMMDR",
    dna: "AAAGGAAACTGGGTCTGCAGGACGTTGCTGCAGTTCAGGCATTGGCCGCGTGGGGGCAGAAGTGCACAGGTCAAAGCGTCTGGAAAAAGCCCTCAGGCTGC",
  },
};

function showElement(node, visible) {
  node.classList.toggle("hidden", !visible);
}

function clearMessages() {
  errorBox.textContent = "";
  infoBox.innerHTML = "";
  showElement(errorBox, false);
  showElement(infoBox, false);
}

function setError(message) {
  errorBox.textContent = message;
  showElement(errorBox, true);
}

function setInfo(messages) {
  if (!messages || messages.length === 0) {
    infoBox.innerHTML = "";
    showElement(infoBox, false);
    return;
  }
  infoBox.innerHTML = messages.map((message) => `<div>${message}</div>`).join("");
  showElement(infoBox, true);
}

function scoreToColor(score) {
  const alpha = 0.08 + (Math.max(0, Math.min(1, score)) * 0.92);
  return `rgba(13, 92, 99, ${alpha.toFixed(3)})`;
}

function getTopPositions(sequence, rawScores, count) {
  return rawScores
    .map((score, index) => ({ score, index, char: sequence[index] }))
    .sort((left, right) => Math.abs(right.score) - Math.abs(left.score))
    .slice(0, count);
}

function buildSummaryLines(data) {
  const summary = data.input_summary;
  const lines = [
    `Original DNA length: ${summary.original_dna_length} bp`,
    `Normalized DNA length: ${summary.normalized_dna_length} bp`,
    `Original protein length: ${summary.original_protein_length} aa`,
    `Normalized protein length: ${summary.normalized_protein_length} aa`,
  ];
  if (!summary.messages.length) {
    lines.push("No input normalization changes were needed.");
  }
  inputSummary.innerHTML = lines.map((line) => `<div>${line}</div>`).join("");
}

function buildNormalizedSummary(data) {
  normalizedSummary.innerHTML = `
    <div class="normalized-block">
      <label>Normalized DNA sequence</label>
      <code>${data.normalized_dna_sequence}</code>
    </div>
    <div class="normalized-block">
      <label>Normalized protein sequence</label>
      <code>${data.normalized_protein_sequence}</code>
    </div>
  `;
}

function buildPeakSummary(data) {
  const dnaTop = getTopPositions(data.normalized_dna_sequence, data.dna_importance_raw, 3);
  const proteinTop = getTopPositions(data.normalized_protein_sequence, data.protein_importance_raw, 3);

  const dnaRows = dnaTop
    .map(
      (item) => `
        <div class="peak-row">
          <span>DNA peak</span>
          <strong>${item.char}${item.index + 1}</strong>
          <span>${item.score.toFixed(6)}</span>
        </div>
      `
    )
    .join("");

  const proteinRows = proteinTop
    .map(
      (item) => `
        <div class="peak-row">
          <span>Protein peak</span>
          <strong>${item.char}${item.index + 1}</strong>
          <span>${item.score.toFixed(6)}</span>
        </div>
      `
    )
    .join("");

  peakSummary.innerHTML = dnaRows + proteinRows;
}

function renderSequence(container, sequence, rawScores, normalizedScores, chunkSize, labelName) {
  container.innerHTML = "";
  for (let start = 0; start < sequence.length; start += chunkSize) {
    const line = document.createElement("div");
    line.className = "sequence-line";

    const lineIndex = document.createElement("span");
    lineIndex.className = "sequence-index";
    lineIndex.textContent = `${start + 1}`;
    line.appendChild(lineIndex);

    const chunk = sequence.slice(start, start + chunkSize);
    for (let index = 0; index < chunk.length; index += 1) {
      const absoluteIndex = start + index;
      const char = document.createElement("span");
      const normScore = normalizedScores[absoluteIndex] ?? 0;
      const rawScore = rawScores[absoluteIndex] ?? 0;
      char.className = "sequence-char";
      char.textContent = chunk[index];
      char.style.backgroundColor = scoreToColor(normScore);
      char.title = `Position: ${absoluteIndex + 1}\n${labelName}: ${chunk[index]}\nRaw score: ${rawScore.toFixed(6)}\nNormalized: ${normScore.toFixed(6)}`;
      line.appendChild(char);
    }

    container.appendChild(line);
  }
}

async function submitPrediction(event) {
  event.preventDefault();
  clearMessages();

  const dnaSequence = dnaInput.value.trim();
  const proteinSequence = proteinInput.value.trim();

  if (!dnaSequence || !proteinSequence) {
    setError("Both protein and DNA sequences are required.");
    return;
  }

  predictButton.disabled = true;
  showElement(loadingIndicator, true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dna_sequence: dnaSequence,
        protein_sequence: proteinSequence,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "Prediction failed.";
      throw new Error(detail);
    }

    emptyState.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    probabilityValue.textContent = data.probability.toFixed(6);
    classValue.textContent = `${data.predicted_class_text} (${data.predicted_label})`;
    logitValue.textContent = data.logit.toFixed(6);

    buildSummaryLines(data);
    buildNormalizedSummary(data);
    buildPeakSummary(data);
    setInfo(data.input_summary.messages);

    renderSequence(
      dnaSequenceView,
      data.normalized_dna_sequence,
      data.dna_importance_raw,
      data.dna_importance_norm,
      101,
      "Base"
    );
    renderSequence(
      proteinSequenceView,
      data.normalized_protein_sequence,
      data.protein_importance_raw,
      data.protein_importance_norm,
      50,
      "Residue"
    );
  } catch (error) {
    setError(error.message || "Prediction failed.");
  } finally {
    predictButton.disabled = false;
    showElement(loadingIndicator, false);
  }
}

form.addEventListener("submit", submitPrediction);

exampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const example = EXAMPLES[button.dataset.example];
    if (!example) {
      return;
    }
    proteinInput.value = example.protein;
    dnaInput.value = example.dna;
    clearMessages();

    const autoRun = button.dataset.autoRun === "true";
    if (autoRun) {
      form.requestSubmit();
    }
  });
});
