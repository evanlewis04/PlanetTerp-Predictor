const fs = require("node:fs");

function loadPlaywright() {
  try {
    return require("playwright");
  } catch (error) {
    throw new Error(
      "Playwright is required for the dashboard smoke test. " +
        "Set NODE_PATH to a node_modules directory that contains playwright, " +
        "or install it as a dev dependency.",
    );
  }
}

const dashboardUrl = process.env.DASHBOARD_URL || "http://127.0.0.1:5173";
const edgePath =
  process.env.EDGE_PATH || "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";

async function main() {
  if (!fs.existsSync(edgePath)) {
    throw new Error(`Microsoft Edge executable was not found at ${edgePath}`);
  }

  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({
    headless: true,
    executablePath: edgePath,
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const consoleErrors = [];
  const badResponses = [];

  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400) {
      badResponses.push({ status: response.status(), url: response.url() });
    }
  });

  await page.goto(dashboardUrl, { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "Overview", exact: true }).waitFor({ timeout: 10000 });
  await page.getByText("Best R2 Model").waitFor({ timeout: 10000 });
  await page.getByText("Random Forest").waitFor({ timeout: 10000 });

  for (const tabName of ["Dataset", "Models", "Features", "Plots", "Predict", "Train", "Runs", "Overview"]) {
    await page.getByRole("button", { name: tabName, exact: true }).click();
    await page.getByRole("heading", { name: tabName, exact: true }).waitFor({ timeout: 10000 });
  }

  await page.getByRole("button", { name: "Train", exact: true }).click();
  await page.getByLabel("Experiment Name").fill("smoke-no-submit");
  await page.getByText("Live API", { exact: true }).click();
  await page.getByRole("button", { name: "Start Training", exact: true }).waitFor({ timeout: 10000 });

  await page.getByRole("button", { name: "Predict", exact: true }).click();
  await page
    .getByPlaceholder("Clear lectures, fair exams, and helpful feedback...")
    .fill("Clear lectures, fair exams, helpful feedback, and manageable homework.");
  await page.getByRole("button", { name: "Predict Rating", exact: true }).click();
  await page.getByText("Missing Filled").waitFor({ timeout: 10000 });
  const predictionText = await page.locator(".rating-meter span").textContent();

  await page.setViewportSize({ width: 390, height: 900 });
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.getByRole("heading", { name: "Overview", exact: true }).waitFor({ timeout: 10000 });
  await page.getByText("Best R2 Model").waitFor({ timeout: 10000 });

  await browser.close();

  if (badResponses.length > 0 || consoleErrors.length > 0) {
    throw new Error(
      JSON.stringify(
        {
          badResponses,
          consoleErrors,
        },
        null,
        2,
      ),
    );
  }

  console.log(
    JSON.stringify(
      {
        dashboardUrl,
        predictionText,
        overviewVisibleMobile: true,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
