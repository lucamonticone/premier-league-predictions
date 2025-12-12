const state = {
  meta: null,
  teams: [],
  sortedTeamIds: [],
  selectedTeamId: null,
  mainChartInitialized: false,
  detailChartInitialized: false,
  selectedModel: "student_t",
  selectedNSim: 2000,
  teamMetaById: {},
  fixturesMeta: null,
  fixtures: []
};

let modelSelectEl = null;
let fixturesToggleButtons = [];


const NAME_TO_CODE = {
  "Arsenal": "ARS",
  "Aston Villa": "AVL",
  "Bournemouth": "BOU",
  "Brentford": "BRE",
  "Brighton Hove": "BHA",
  "Burnley": "BUR",
  "Chelsea": "CHE",
  "Crystal Palace": "CRY",
  "Everton": "EVE",
  "Fulham": "FUL",
  "Leeds United": "LEE",
  "Liverpool": "LIV",
  "Man City": "MCI",
  "Man United": "MUN",
  "Newcastle": "NEW",
  "Nottingham": "NFO",
  "Sunderland": "SUN",
  "Tottenham": "TOT",
  "West Ham": "WHU",
  "Wolverhampton": "WOL"
};

const LOGO_SIZE_NORMAL = 8.5;
const LOGO_SIZE_SELECTED = 10;

const loadJson = async (path) => {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Error loading ${path}: ${res.status}`);
  }
  return res.json();
};

const buildState = (teamMeta, outcomes) => {
  const byId = {};

  teamMeta.forEach((t) => {
    if (t.team_id) byId[t.team_id] = t;
    if (t.name) byId[t.name] = t;
    if (t.short_name) byId[t.short_name] = t;
  });

  state.teamMetaById = byId;
  state.meta = outcomes.meta || null;

  state.teams = (outcomes.teams || []).map((t) => {
    const longName = t.team_id;
    const code = NAME_TO_CODE[longName] || longName;
    const meta = byId[code] || byId[longName] || {};

    const rawQ = t.points_quantiles || {};
    const q = {
      p05: rawQ.p05 ?? rawQ.q05 ?? null,
      p25: rawQ.p25 ?? rawQ.q25 ?? null,
      p50: rawQ.p50 ?? rawQ.q50 ?? null,
      p75: rawQ.p75 ?? rawQ.q75 ?? null,
      p95: rawQ.p95 ?? rawQ.q95 ?? null
    };

    const rank = (t.probabilities && t.probabilities.rank) || {};
    const r = (k) => (k in rank ? rank[k] : 0);

    const probsAgg = {
      champion: r("1"),
      top4: r("1") + r("2") + r("3") + r("4"),
      top6:
        r("1") +
        r("2") +
        r("3") +
        r("4") +
        r("5") +
        r("6"),
      relegation: r("18") + r("19") + r("20")
    };

    return {
      team_id: longName,
      name: meta.name || longName,
      short_name: meta.short_name || longName,
      logo: meta.logo || null,
      primary_color: meta.primary_color || "#2563eb",
      secondary_color: meta.secondary_color || "#e5e7eb",
      stats: {
        mean_points: t.mean_points,
        sd_points: t.sd_points,
        points_quantiles: q,
        probabilities: probsAgg,
        most_likely_rank: t.most_likely_rank
      }
    };
  });
};

const formatPct = (p) => {
  if (p == null) return "-";
  const v = p * 100;
  if (v >= 10) return `${v.toFixed(0)}%`;
  if (v >= 1) return `${v.toFixed(1)}%`;
  return `< ${v.toFixed(1)}%`;
};

const formatPct1 = (p) => {
  if (p == null) return "-";
  return `${(p * 100).toFixed(1)}%`;
};

const gaussianSample = (mean, sd) => {
  let u = 1 - Math.random();
  let v = 1 - Math.random();
  let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return mean + sd * z;
};

const buildMainLayouts = (teamsSorted, xs, ys, q05, q95, baselineY) => {
  const n = teamsSorted.length;
  const yMax = Math.max(...q95);

  const xStart = 0;
  const baseX = new Array(n).fill(xStart);
  const baseY = new Array(n).fill(baselineY);

  const makeImages = (xVals, yVals) =>
    teamsSorted.map((t, i) => {
      const isSelected = t.team_id === state.selectedTeamId;
      const size = isSelected ? LOGO_SIZE_SELECTED : LOGO_SIZE_NORMAL;

      return {
        source: t.logo || "",
        xref: "x",
        yref: "y",
        x: xVals[i],
        y: yVals[i],
        sizex: size,
        sizey: size,
        xanchor: "center",
        yanchor: "middle",
        layer: "above"
      };
    });

  const baseImages = makeImages(baseX, baseY);
  const finalImages = makeImages(xs, ys);

  const common = {
    margin: { l: 40, r: 10, t: 10, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {
      range: [-0.5, n + 1],
      tickmode: "array",
      tickvals: xs,
      ticktext: teamsSorted.map((t) => t.short_name),
      tickangle: -40,
      fixedrange: true
    },
    yaxis: {
      title: "Projected final points",
      range: [baselineY, yMax + 5],
      fixedrange: true,
      zeroline: false
    },
    hovermode: "closest",
    showlegend: false
  };

  const layoutBase = {
    ...common,
    images: baseImages
  };

  const layoutFinal = {
    ...common,
    images: finalImages
  };

  return { layoutBase, layoutFinal };
};

const renderMainChart = () => {
  if (!state.teams.length) return;

  const teams = [...state.teams].sort(
    (a, b) => b.stats.mean_points - a.stats.mean_points
  );
  state.sortedTeamIds = teams.map((t) => t.team_id);

  const n = teams.length;
  const xs = teams.map((_, i) => i + 1);
  const ys = teams.map((t) => t.stats.mean_points);
  const q05 = teams.map((t) => t.stats.points_quantiles.p05);
  const q95 = teams.map((t) => t.stats.points_quantiles.p95);

  const yMin = Math.min(...q05);
  const baselineY = yMin - 5;

  const { layoutBase, layoutFinal } = buildMainLayouts(
    teams,
    xs,
    ys,
    q05,
    q95,
    baselineY
  );

  const tracesBase = [];
  const tracesFinal = [];

  for (let i = 0; i < n; i++) {
    const t = teams[i];
    const color = t.primary_color;
    const mean = ys[i];
    const yLow = q05[i];
    const yHigh = q95[i];

    const markerSize = 26;
    const customData = [[t.name, yLow, yHigh]];

    const baseBorder = {
      x: [xs[i], xs[i]],
      y: [baselineY, baselineY],
      type: "scatter",
      mode: "lines",
      line: {
        color: "#111827",
        width: 0
      },
      hoverinfo: "skip",
      showlegend: false
    };

    const baseColor = {
      x: [xs[i], xs[i]],
      y: [baselineY, baselineY],
      type: "scatter",
      mode: "lines",
      line: {
        color: color,
        width: 0
      },
      hoverinfo: "skip",
      showlegend: false
    };

    const baseMarker = {
      x: [xs[i]],
      y: [baselineY],
      type: "scatter",
      mode: "markers",
      marker: {
        size: markerSize,
        color: "rgba(0,0,0,0)",
        opacity: 0
      },
      customdata: customData,
      hovertemplate:
        "<b>%{customdata[0]}</b><br>" +
        "Mean points: %{y:.1f}<br><extra></extra>",
      showlegend: false
    };

    const finalBorder = {
      x: [xs[i], xs[i]],
      y: [yLow, yHigh],
      type: "scatter",
      mode: "lines",
      line: {
        color: "#111827",
        width: 30
      },
      hoverinfo: "skip",
      showlegend: false
    };

    const finalColor = {
      x: [xs[i], xs[i]],
      y: [yLow, yHigh],
      type: "scatter",
      mode: "lines",
      line: {
        color: color,
        width: 25
      },
      hoverinfo: "skip",
      showlegend: false
    };

    const finalMarker = {
      x: [xs[i]],
      y: [mean],
      type: "scatter",
      mode: "markers",
      marker: {
        size: markerSize,
        color: "rgba(0,0,0,0)",
        opacity: 0
      },
      customdata: customData,
      hovertemplate:
        "<b>%{customdata[0]}</b><br>" +
        "Mean points: %{y:.1f}<br>" +
        "5–95%: %{customdata[1]}–%{customdata[2]}<extra></extra>",
      showlegend: false
    };

    tracesBase.push(baseBorder, baseColor, baseMarker);
    tracesFinal.push(finalBorder, finalColor, finalMarker);
  }

  const config = {
    displayModeBar: false,
    responsive: true
  };

  const chartEl = document.getElementById("points-chart");

  if (!state.mainChartInitialized) {
    state.mainChartInitialized = true;

    Plotly.newPlot(chartEl, tracesBase, layoutBase, config).then(() => {
      chartEl.on("plotly_click", handleMainChartClick);

      Plotly.animate(
        chartEl,
        {
          data: tracesFinal,
          layout: layoutFinal
        },
        {
          transition: {
            duration: 1100,
            easing: "cubic-in-out"
          },
          frame: {
            duration: 1100,
            redraw: false
          }
        }
      );
    });
  } else {
    Plotly.react(chartEl, tracesFinal, layoutFinal, config);
  }
};

const handleMainChartClick = (evt) => {
  if (!evt.points || !evt.points.length) return;
  const curveIndex = evt.points[0].curveNumber;
  const teamIndex = Math.floor(curveIndex / 3);
  const teamId = state.sortedTeamIds[teamIndex];
  if (teamId) {
    selectTeam(teamId);
  }
};

const renderDetailChart = (team) => {
  const targetEl = document.getElementById("detail-points-chart");
  if (!team) {
    if (state.detailChartInitialized) {
      Plotly.purge(targetEl);
      state.detailChartInitialized = false;
    }
    return;
  }

  const mean = team.stats.mean_points;
  const sd = team.stats.sd_points || 5;

  const samples = [];
  const n = 400;
  for (let i = 0; i < n; i++) {
    let s = gaussianSample(mean, sd);
    s = Math.max(20, Math.min(100, s));
    samples.push(s);
  }

  const trace = {
    x: samples,
    type: "histogram",
    nbinsx: 18,
    marker: {
      color: team.primary_color,
      opacity: 0.8,
      line: {
        color: "rgba(15,23,42,0.15)",
        width: 1
      }
    },
    hovertemplate: "Points: %{x:.0f}<br>Simulated count: %{y}<extra></extra>"
  };

  const layout = {
    margin: { l: 40, r: 10, t: 10, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {
      title: "Points",
      fixedrange: true
    },
    yaxis: {
      title: "Simulated seasons",
      fixedrange: true
    },
    bargap: 0.05
  };

  const config = {
    displayModeBar: false,
    responsive: true
  };

  if (!state.detailChartInitialized) {
    Plotly.newPlot(targetEl, [trace], layout, config);
    state.detailChartInitialized = true;
  } else {
    Plotly.react(targetEl, [trace], layout, config);
  }
};

const selectTeam = (teamId) => {
  const team = state.teams.find((t) => t.team_id === teamId);
  if (!team) return;

  state.selectedTeamId = teamId;

  const placeholder = document.getElementById("detail-placeholder");
  const content = document.getElementById("detail-content");
  placeholder.classList.add("hidden");
  content.classList.remove("hidden");

  const badge = document.getElementById("detail-team-badge");
  badge.innerHTML = "";
  if (team.logo) {
    const img = document.createElement("img");
    img.src = team.logo;
    img.alt = team.name;
    badge.style.background = "#ffffff";
    badge.appendChild(img);
  } else {
    badge.style.background = team.primary_color;
    badge.textContent = team.short_name;
  }

  const titleEl = document.getElementById("detail-team-name");
  const subtitleEl = document.getElementById("detail-team-subtitle");

  titleEl.textContent = team.name;
  subtitleEl.textContent = `${team.stats.mean_points.toFixed(
    1
  )} pts • most likely rank ${team.stats.most_likely_rank}`;

  const probs = team.stats.probabilities || {};
  document.getElementById("prob-champion").textContent = formatPct(
    probs.champion
  );
  document.getElementById("prob-top4").textContent = formatPct(probs.top4);
  document.getElementById("prob-top6").textContent = formatPct(probs.top6);
  document.getElementById("prob-relegation").textContent = formatPct(
    probs.relegation
  );

  const q = team.stats.points_quantiles;
  document.getElementById("summary-mean-points").textContent =
    team.stats.mean_points.toFixed(1);
  document.getElementById(
    "summary-range-points"
  ).textContent = `${q.p05}–${q.p95}`;
  document.getElementById(
    "summary-most-likely-rank"
  ).textContent = `#${team.stats.most_likely_rank}`;

  renderDetailChart(team);
  renderMainChart();
};

const buildFixturesState = (predictions) => {
  if (!predictions) {
    state.fixturesMeta = null;
    state.fixtures = [];
    return;
  }

  state.fixturesMeta = {
    season: predictions.season,
    model: predictions.model,
    matchweek: predictions.matchweek,
    last_update: predictions.last_update
  };

  const fixtures = predictions.fixtures || [];
  const byId = state.teamMetaById || {};

  state.fixtures = fixtures.map((f) => {
    const codeH = NAME_TO_CODE[f.home_team] || f.home_team;
    const codeA = NAME_TO_CODE[f.away_team] || f.away_team;

    const hMeta = byId[codeH] || byId[f.home_team] || {};
    const aMeta = byId[codeA] || byId[f.away_team] || {};

    const home = {
      team_id: f.home_team,
      name: hMeta.name || f.home_team,
      short_name: hMeta.short_name || f.home_team,
      logo: hMeta.logo || null,
      primary_color: hMeta.primary_color || "#ec4899",
      secondary_color: hMeta.secondary_color || "#fee2e2"
    };

    const away = {
      team_id: f.away_team,
      name: aMeta.name || f.away_team,
      short_name: aMeta.short_name || f.away_team,
      logo: aMeta.logo || null,
      primary_color: aMeta.primary_color || "#22c55e",
      secondary_color: aMeta.secondary_color || "#dcfce7"
    };

    return {
      home,
      away,
      p_home: f.p_home,
      p_draw: f.p_draw,
      p_away: f.p_away,
      exp_home_goals: f.exp_home_goals,
      exp_away_goals: f.exp_away_goals
    };
  });
};

const renderFixturesGrid = () => {
  const grid = document.getElementById("fixtures-grid");
  const subtitle = document.getElementById("fixtures-subtitle");
  if (!grid) return;

  grid.innerHTML = "";

  if (!state.fixtures || !state.fixtures.length) {
    if (subtitle) {
      subtitle.textContent = "No upcoming fixtures available.";
    }
    return;
  }

  if (subtitle && state.fixturesMeta) {
    const { matchweek, last_update } = state.fixturesMeta;
    let when = "";
    if (last_update) {
      const d = new Date(last_update);
      if (!Number.isNaN(d.getTime())) {
        when = d.toLocaleString(undefined, {
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          hour12: false
        });
      }
    }
    subtitle.textContent = when
      ? `Matchweek ${matchweek} • updated ${when}`
      : `Matchweek ${matchweek}`;
  }

  const makeTeamNode = (side, info) => {
    const wrap = document.createElement("div");
    wrap.className = `fixture-team fixture-team--${side}`;

    const logoWrap = document.createElement("div");
    logoWrap.className = "fixture-logo";
    logoWrap.style.background = info.secondary_color || "#e5e7eb";

    if (info.logo) {
      const img = document.createElement("img");
      img.src = info.logo;
      img.alt = info.name;
      logoWrap.appendChild(img);
    } else {
      logoWrap.textContent = info.short_name.slice(0, 3).toUpperCase();
    }

    const nameSpan = document.createElement("span");
    nameSpan.className = "fixture-team-name";
    nameSpan.textContent = info.short_name;

    wrap.appendChild(logoWrap);
    wrap.appendChild(nameSpan);

    return wrap;
  };

  state.fixtures.forEach((f) => {
    const card = document.createElement("div");
    card.className = "fixture-card";

    const header = document.createElement("div");
    header.className = "fixture-header";

    const homeNode = makeTeamNode("home", f.home);
    const awayNode = makeTeamNode("away", f.away);

    const vs = document.createElement("span");
    vs.className = "fixture-vs";
    vs.textContent = "vs";

    header.appendChild(homeNode);
    header.appendChild(vs);
    header.appendChild(awayNode);

    const bar = document.createElement("div");
    bar.className = "fixture-bar";

    const segHome = document.createElement("div");
    segHome.className = "fixture-bar__segment fixture-bar__home";
    segHome.style.flexGrow = f.p_home || 0;
    segHome.style.background = f.home.primary_color;

    const segDraw = document.createElement("div");
    segDraw.className = "fixture-bar__segment fixture-bar__draw";
    segDraw.style.flexGrow = f.p_draw || 0;

    const segAway = document.createElement("div");
    segAway.className = "fixture-bar__segment fixture-bar__away";
    segAway.style.flexGrow = f.p_away || 0;
    segAway.style.background = f.away.primary_color;

    bar.appendChild(segHome);
    bar.appendChild(segDraw);
    bar.appendChild(segAway);

    const probRow = document.createElement("div");
    probRow.className = "fixture-prob-row";
    probRow.innerHTML = `
      <span class="fixture-prob fixture-prob--home">${formatPct1(
        f.p_home
      )}</span>
      <span class="fixture-prob fixture-prob--draw">Draw ${formatPct1(
        f.p_draw
      )}</span>
      <span class="fixture-prob fixture-prob--away">${formatPct1(
        f.p_away
      )}</span>
    `;

    const xgRow = document.createElement("div");
    xgRow.className = "fixture-xg-row";
    xgRow.innerHTML = `
      <span>${f.home.short_name} xG ${f.exp_home_goals.toFixed(2)}</span>
      <span>${f.away.short_name} xG ${f.exp_away_goals.toFixed(2)}</span>
    `;

    card.appendChild(header);
    card.appendChild(bar);
    card.appendChild(probRow);
    card.appendChild(xgRow);

    grid.appendChild(card);
  });
};

const syncFixturesToggle = () => {
  if (!fixturesToggleButtons || !fixturesToggleButtons.length) return;
  fixturesToggleButtons.forEach((btn) => {
    const model = btn.dataset.modelToggle;
    btn.classList.toggle("chip-button--active", model === state.selectedModel);
  });
};


const getOutcomesPath = () => {
  const model = state.selectedModel;
  const nSim = state.selectedNSim;
  return `data/premier_outcomes_${model}_${nSim}.json`;
};

const loadDataAndRender = async () => {
  try {
    const [teamMeta, outcomes] = await Promise.all([
      loadJson("data/team_metadata.json"),
      loadJson(getOutcomesPath())
    ]);

    buildState(teamMeta, outcomes);

    let predictions = null;
    try {
      const modelPath = `data/matchday_predictions_${state.selectedModel}.json`;
      predictions = await loadJson(modelPath);
    } catch (e1) {
      try {
        predictions = await loadJson("data/matchday_predictions.json");
      } catch (e2) {
        console.warn("No matchday predictions file found", e2);
      }
    }

    buildFixturesState(predictions);
    renderMainChart();
    renderFixturesGrid();
    syncFixturesToggle();      // <<<<<< aggiunta
  } catch (err) {
    console.error(err);
    showError("Unable to load outcomes data. Check console for details.");
  }
};


const wireControls = () => {
  modelSelectEl = document.getElementById("model-select");
  const nsimSelect = document.getElementById("nsim-select");
  const runButton = document.getElementById("run-button");
  fixturesToggleButtons = Array.from(
    document.querySelectorAll("[data-model-toggle]")
  );

  if (modelSelectEl) {
    state.selectedModel = modelSelectEl.value;
    modelSelectEl.addEventListener("change", (e) => {
      state.selectedModel = e.target.value;
      syncFixturesToggle();
    });
  }

  if (nsimSelect) {
    state.selectedNSim = Number(nsimSelect.value);
    nsimSelect.addEventListener("change", (e) => {
      state.selectedNSim = Number(e.target.value);
    });
  }

  if (runButton) {
    runButton.addEventListener("click", () => {
      loadDataAndRender();
    });
  }

  fixturesToggleButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const model = btn.dataset.modelToggle;
      if (!model) return;
      state.selectedModel = model;
      if (modelSelectEl) modelSelectEl.value = model;
      syncFixturesToggle();
      loadDataAndRender();
    });
  });

  syncFixturesToggle();
};


const showError = (msg) => {
  const root = document.querySelector(".app-root");
  if (!root) return;
  const div = document.createElement("div");
  div.textContent = msg;
  div.style.marginTop = "8px";
  div.style.fontSize = "0.85rem";
  div.style.color = "#b91c1c";
  root.appendChild(div);
};

const initApp = async () => {
  wireControls();
  await loadDataAndRender();
};

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});
