// Backend API endpoint
const API_URL = "http://localhost:8000/predict"; // Update if backend runs on a different port

const form = document.getElementById("loanForm");
const steps = document.querySelectorAll(".step");
const progressSteps = document.querySelectorAll(".progress-step");
const progressFill = document.querySelector(".progress-fill");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const resultBox = document.getElementById("result");
const riskScoreEl = document.getElementById("riskScore");
const decisionEl = document.getElementById("decision");
const meterCircle = document.querySelector(".meter-circle");
const resetBtn = document.getElementById("resetBtn");

let currentStep = 0;

function showStep(step) {
  steps.forEach((s, i) => {
    s.classList.toggle("active", i === step);
  });
  progressSteps.forEach((ps, i) => {
    ps.classList.toggle("active", i === step);
    ps.classList.toggle("completed", i < step);
  });
  progressFill.style.width = `${(step / (steps.length - 1)) * 100}%`;

  prevBtn.disabled = step === 0;
  nextBtn.textContent = step === steps.length - 1 ? "Check Risk" : "Next";
  nextBtn.disabled = false;
}

function validateStep(step) {
  const inputs = steps[step].querySelectorAll("input, select");
  let valid = true;
  inputs.forEach(input => {
    if (!input.checkValidity()) {
      input.reportValidity();
      valid = false;
    }
  });
  return valid;
}

prevBtn.addEventListener("click", () => {
  currentStep--;
  showStep(currentStep);
});

nextBtn.addEventListener("click", async () => {
  if (!validateStep(currentStep)) return;

  if (currentStep < steps.length - 1) {
    currentStep++;
    showStep(currentStep);
  } else {
    // Submit form
    nextBtn.disabled = true;
    nextBtn.textContent = "Loading...";

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert numeric fields
    for (let key in data) {
      if (!isNaN(data[key]) && data[key] !== "") {
        data[key] = Number(data[key]);
      }
    }

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify(data),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server error: ${res.status} - ${errorText || "Unknown error"}`);
      }

      const result = await res.json();

      // Show result
      resultBox.classList.remove("hidden");
      form.classList.add("hidden");

      const scorePercent = (result.risk_score * 100).toFixed(2);
      riskScoreEl.textContent = `${scorePercent}%`;
      decisionEl.textContent = result.decision;

      // Color coding
      const color = result.decision === "High Risk" ? "#ff5252" : "#00c4b4";
      decisionEl.style.color = color;
      meterCircle.style.background = `conic-gradient(${color} 0% ${scorePercent}%, #e0e7ff ${scorePercent}% 100%)`;

      // Scroll to result
      resultBox.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
      console.error("Error:", err);
      alert(`⚠️ Failed to get prediction: ${err.message}. Please check your input or try again later.`);
    } finally {
      nextBtn.disabled = false;
      nextBtn.textContent = "Check Risk";
    }
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  resultBox.classList.add("hidden");
  form.classList.remove("hidden");
  currentStep = 0;
  showStep(currentStep);
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// Initial show
showStep(currentStep);