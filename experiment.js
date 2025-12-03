// Core experiment logic for probability RL (Liking x Difficulty)
// Stage1: rating; Stage2: fixed 4-round probabilistic RL (all 6 groups covered at least once)

const config = {
  responseKeys: {
    left: ["ArrowLeft", "a", "A"],
    right: ["ArrowRight", "l", "L"]
  },
  groups: [
    { id: "G1", label: "puppies", liking: "high", images: [
      "images(1)/images/puppies/download.jpg",
      "images(1)/images/puppies/images (1).jpg",
      "images(1)/images/puppies/images (2).jpg",
      "images(1)/images/puppies/images (3).jpg",
      "images(1)/images/puppies/images (4).jpg",
      "images(1)/images/puppies/images (5).jpg",
      "images(1)/images/puppies/images (6).jpg",
      "images(1)/images/puppies/images 3.jpg",
      "images(1)/images/puppies/images.jpg",
      "images(1)/images/puppies/images1.jpg"
    ] },
    { id: "G2", label: "nature", liking: "high", images: [
      "images(1)/images/natrure/7.jpg",
      "images(1)/images/natrure/download (1).jpg",
      "images(1)/images/natrure/download (2).jpg",
      "images(1)/images/natrure/download.jpg",
      "images(1)/images/natrure/images (1).jpg",
      "images(1)/images/natrure/images (2).jpg",
      "images(1)/images/natrure/images (3).jpg",
      "images(1)/images/natrure/images (4).jpg",
      "images(1)/images/natrure/images (6).jpg",
      "images(1)/images/natrure/images.jpg"
    ] },
    { id: "G3", label: "babies", liking: "high", images: [
      "images(1)/images/babies/download (3).jpg",
      "images(1)/images/babies/download.jpg",
      "images(1)/images/babies/images (1).jpg",
      "images(1)/images/babies/images (2).jpg",
      "images(1)/images/babies/images (3).jpg",
      "images(1)/images/babies/images (4).jpg",
      "images(1)/images/babies/images (5).jpg",
      "images(1)/images/babies/images (7).jpg",
      "images(1)/images/babies/images.jpg",
      "images(1)/images/babies/images1.jpg"
    ] },
    { id: "G4", label: "alcohol", liking: "low", images: [
      "images(1)/images/alchohol/1.jpg",
      "images(1)/images/alchohol/download (1).jpg",
      "images(1)/images/alchohol/download.jpg",
      "images(1)/images/alchohol/images (1).jpg",
      "images(1)/images/alchohol/images (2).jpg",
      "images(1)/images/alchohol/images (3).jpg",
      "images(1)/images/alchohol/images (4).jpg",
      "images(1)/images/alchohol/images (5).jpg",
      "images(1)/images/alchohol/images (6).jpg",
      "images(1)/images/alchohol/images.jpg"
    ] },
    { id: "G5", label: "neutral", liking: "low", images: [
      "images(1)/images/neutral/download (1).jpg",
      "images(1)/images/neutral/download (2).jpg",
      "images(1)/images/neutral/download (3).jpg",
      "images(1)/images/neutral/download (4).jpg",
      "images(1)/images/neutral/download (5).jpg",
      "images(1)/images/neutral/download (6).jpg",
      "images(1)/images/neutral/download (7).jpg",
      "images(1)/images/neutral/download.jpg",
      "images(1)/images/neutral/images (1).jpg",
      "images(1)/images/neutral/images.jpg"
    ] },
    { id: "G6", label: "negative", liking: "low", images: [
      "images(1)/images/negative/4.jpg",
      "images(1)/images/negative/download (1).jpg",
      "images(1)/images/negative/download (2).jpg",
      "images(1)/images/negative/download (3).jpg",
      "images(1)/images/negative/download (4).jpg",
      "images(1)/images/negative/download (5).jpg",
      "images(1)/images/negative/download (6).jpg",
      "images(1)/images/negative/download.jpg",
      "images(1)/images/negative/images (1).jpg",
      "images(1)/images/negative/images.jpg"
    ] }
  ],
  // schedule template; groupIds assigned randomly at runtime to ensure all 6 groups appear at least once
  scheduleTemplate: [
    { round: 1, name: "Round 1", pCorrect: 0.9, difficulty: "easy", difficultyRound: 0, nGroups: 3, trialsPerGroup: 30 },
    { round: 2, name: "Round 2", pCorrect: 0.7, difficulty: "hard", difficultyRound: 1, nGroups: 4, trialsPerGroup: 30 }
  ]
};

const state = {
  ratingQueue: [],
  ratingIndex: 0,
  ratings: [],
  ratingLookup: {}, // image -> rating value
  trials: [],
  trialIndex: -1,
  runInfo: null,
  keyMap: {},
  waitingForResponse: false,
  trialStartTime: 0,
  currentRoundIdx: 0,
  schedule: []
};

const $ = (id) => document.getElementById(id);
const shuffle = (arr) => {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

const buildSchedule = () => {
  const allIds = config.groups.map((g) => g.id);
  const template = config.scheduleTemplate;
  const baseOrder = shuffle(allIds); // ensures each group used at least once in first two rounds
  let cursor = 0;
  const schedule = [];
  template.forEach((t, idx) => {
    let groupIds;
    if (idx === 0 || idx === 1) {
      groupIds = [];
      for (let i = 0; i < t.nGroups; i += 1) {
        groupIds.push(baseOrder[(cursor + i) % allIds.length]);
      }
      cursor += t.nGroups;
    } else {
      const pool = shuffle(allIds);
      groupIds = pool.slice(0, t.nGroups);
    }
    schedule.push({ ...t, groupIds });
  });
  return schedule;
};

const statusLine = (msg) => { $("status-line").textContent = `Status: ${msg}`; };
const roundLine = () => {
  const total = state.schedule.length || config.scheduleTemplate.length;
  $("round-line").textContent = `Round: ${Math.min(state.currentRoundIdx + 1, total)} / ${total} (click \"开始当前轮任务\" after finishing to proceed)`;
};

// Build a stable state ID per image for export
const imageStateMap = (() => {
  const map = {};
  let idx = 1;
  config.groups.forEach((g) => {
    g.images.forEach((img) => {
      if (!map[img]) {
        map[img] = idx;
        idx += 1;
      }
    });
  });
  return map;
})();

// ----- Rating stage -----
function buildRatingQueue() {
  state.ratingQueue = [];
  config.groups.forEach((g) => {
    g.images.forEach((img) => {
      state.ratingQueue.push({ groupId: g.id, label: g.label, liking: g.liking, image: img });
    });
  });
  state.ratingQueue = shuffle(state.ratingQueue);
  state.ratingIndex = 0;
}

function showNextRating() {
  if (state.ratingIndex >= state.ratingQueue.length) {
    $("rating-frame").style.display = "none";
    statusLine("Rating finished. You can start the RL task.");
    state.ratingLookup = {};
    state.ratings.forEach((r) => { state.ratingLookup[r.image] = r.rating; });
    return;
  }
  const item = state.ratingQueue[state.ratingIndex];
  $("rating-img-tag").src = item.image;
  $("rating-value").textContent = `当前评分：${$("rating-slider").value}`;
  $("rating-frame").style.display = "block";
  $("rating-instructions").textContent = `进度 ${state.ratingIndex + 1}/${state.ratingQueue.length} - 组别 ${item.groupId} (${item.label})`;
}

function submitRating() {
  if (state.ratingIndex >= state.ratingQueue.length) return;
  const item = state.ratingQueue[state.ratingIndex];
  state.ratings.push({
    participantId: $("participant-id").value || "",
    sessionLabel: $("session-label").value || "",
    groupId: item.groupId,
    groupLabel: item.label,
    likingLabel: item.liking,
    image: item.image,
    rating: Number($("rating-slider").value),
    timestamp: Date.now()
  });
  state.ratingIndex += 1;
  showNextRating();
}

// ----- Trial building -----
function assignKeys(groupIds) {
  const map = {};
  groupIds.forEach((gid) => {
    map[gid] = Math.random() > 0.5 ? "left" : "right";
  });
  return map;
}

function groupById(ids) {
  const lookup = Object.fromEntries(config.groups.map((g) => [g.id, g]));
  return ids.map((id) => lookup[id]).filter(Boolean);
}

function buildTrialsForRound(roundCfg) {
  const groups = groupById(roundCfg.groupIds);
  const trials = [];
  groups.forEach((g) => {
    const perGroup = [];
    for (let i = 0; i < roundCfg.trialsPerGroup; i += 1) {
      const img = g.images[i % g.images.length];
      perGroup.push({
        groupId: g.id,
        groupLabel: g.label,
        liking: g.liking,
        image: img,
        trialInGroup: i + 1
      });
    }
    trials.push(...shuffle(perGroup));
  });
  return shuffle(trials).map((t, idx) => ({
    ...t,
    trialIndex: idx + 1,
    round: roundCfg.round,
    difficulty: roundCfg.difficulty,
    difficultyRound: roundCfg.difficultyRound,
    pCorrect: roundCfg.pCorrect
  }));
}

function likingFromRating(imagePath, defaultLiking) {
  const rating = state.ratingLookup[imagePath];
  if (rating === undefined) {
    return defaultLiking === "high" ? 1 : 0;
  }
  if (rating > 5) return 1;
  if (rating < 5) return 0;
  return ""; // rating == 5, mark empty for filtering
}

function initRound() {
  if (!state.schedule.length) {
    state.schedule = buildSchedule();
  }
  if (state.currentRoundIdx >= state.schedule.length) {
    statusLine("All rounds are finished.");
    return false;
  }
  const participantId = $("participant-id").value.trim();
  if (!participantId) {
    alert("请先输入被试编号 (Participant ID)");
    return false;
  }
  const sessionLabelUser = $("session-label").value.trim() || "session-1";
  const roundCfg = state.schedule[state.currentRoundIdx];
  const keyMap = assignKeys(roundCfg.groupIds);
  const trials = buildTrialsForRound(roundCfg);

  state.runInfo = {
    participantId,
    sessionLabel: `round-${roundCfg.round}`, // export uses internal round tag
    sessionLabelUser,
    round: roundCfg.round,
    difficulty: roundCfg.difficulty,
    difficultyRound: roundCfg.difficultyRound,
    pCorrect: roundCfg.pCorrect
  };
  state.trials = trials;
  state.trialIndex = 0;
  state.keyMap = keyMap;
  state.waitingForResponse = false;
  $("progress-bar").style.width = "0%";
  statusLine(`Round ${roundCfg.round} (${roundCfg.difficulty}, p=${roundCfg.pCorrect}) running...`);
  roundLine();
  return true;
}

// ----- Trial flow -----
function presentTrial() {
  if (state.trialIndex >= state.trials.length) {
    statusLine(`Round ${state.runInfo.round} finished. Download or start next round.`);
    $("feedback").textContent = "本轮结束 / Session complete.";
    $("stimulus-img").src = "";
    state.currentRoundIdx += 1;
    roundLine();
    return;
  }
  const t = state.trials[state.trialIndex];
  $("stimulus-img").src = t.image;
  $("feedback").textContent = "";
  $("trial-meta").textContent = `Round ${t.round} | Trial ${t.trialIndex}/${state.trials.length} | Group ${t.groupId} (${t.groupLabel}, ${t.liking}) | Difficulty ${t.difficulty}`;
  const pct = Math.round((t.trialIndex - 1) / state.trials.length * 100);
  $("progress-bar").style.width = `${pct}%`;
  state.trialStartTime = performance.now();
  state.waitingForResponse = true;
}

function handleChoice(side, source = "key") {
  if (!state.waitingForResponse || state.trialIndex >= state.trials.length) return;
  const t = state.trials[state.trialIndex];
  const assigned = state.keyMap[t.groupId];
  const isCorrectChoice = side === assigned;
  const pCorrect = t.pCorrect;
  const feedbackPositive = Math.random() < (isCorrectChoice ? pCorrect : 1 - pCorrect);
  const rt = Math.round(performance.now() - state.trialStartTime);

  const likingBin = likingFromRating(t.image, t.liking);

  state.trials[state.trialIndex] = {
    ...t,
    responseSide: side,
    responseSource: source,
    assignedSide: assigned,
    isCorrectChoice,
    feedbackPositive,
    rtMs: rt,
    timestamp: Date.now(),
    stateId: imageStateMap[t.image],
    action: side === "left" ? 0 : 1,
    reward: feedbackPositive ? 1 : 0,
    likingBin
  };
  $("feedback").textContent = feedbackPositive ? "正确 / Correct" : "错误 / Wrong";
  state.waitingForResponse = false;

  setTimeout(() => {
    state.trialIndex += 1;
    presentTrial();
  }, 700);
}

function startExperiment() {
  if (!initRound()) return;
  presentTrial();
}

// ----- Data export -----
function toCSV(rows) {
  if (!rows || rows.length === 0) return "";
  const headers = Object.keys(rows[0]);
  const escape = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };
  const lines = [headers.join(",")];
  rows.forEach((r) => lines.push(headers.map((h) => escape(r[h])).join(",")));
  return lines.join("\n");
}

function downloadFile(content, filename) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadData() {
  if (state.trials.length === 0) {
    alert("没有 trial 数据可导出。请先运行实验。");
    return;
  }
  const baseRow = () => ({
    record_type: "",
    subj_id: state.runInfo.participantId,
    sessionLabel: state.runInfo.sessionLabel,
    sessionLabelUser: state.runInfo.sessionLabelUser || "",
    round: "",
    trial_in_round: "",
    difficulty_round: "",
    difficulty: "",
    state: "",
    groupId: "",
    groupLabel: "",
    liking_group: "",
    liking: "",
    image: "",
    assignedSide: "",
    action: "",
    responseSide: "",
    responseSource: "",
    isCorrectChoice: "",
    reward: "",
    feedbackPositive: "",
    rtMs: "",
    rating: "",
    timestamp: ""
  });

  const behaviorRows = state.trials
    .filter((t) => t.responseSide)
    .map((t) => ({
      ...baseRow(),
      record_type: "behavior",
      round: t.round,
      trial_in_round: t.trialIndex,
      difficulty_round: state.runInfo.difficultyRound,
      difficulty: t.difficulty,
      state: t.stateId,
      groupId: t.groupId,
      groupLabel: t.groupLabel,
      liking_group: t.liking,
      liking: t.likingBin,
      image: t.image,
      assignedSide: t.assignedSide,
      action: t.action,
      responseSide: t.responseSide,
      responseSource: t.responseSource,
      isCorrectChoice: t.isCorrectChoice ? 1 : 0,
      reward: t.reward,
      feedbackPositive: t.feedbackPositive ? 1 : 0,
      rtMs: t.rtMs,
      timestamp: t.timestamp
    }));

  const ratingRows = state.ratings.map((r) => ({
    ...baseRow(),
    record_type: "rating",
    liking_group: r.likingLabel,
    liking: r.rating,
    image: r.image,
    groupId: r.groupId,
    groupLabel: r.groupLabel,
    timestamp: r.timestamp
  }));

  const allRows = [...behaviorRows, ...ratingRows];
  if (allRows.length > 0) {
    downloadFile(toCSV(allRows), `subj-${state.runInfo.participantId}-all.csv`);
    statusLine("数据已导出（单一合并文件）。");
  } else {
    alert("暂无可导出的数据。");
  }
}

// ----- Event wiring -----
function setup() {
  state.schedule = buildSchedule();
  $("start-rating").addEventListener("click", () => {
    buildRatingQueue();
    statusLine("Rating in progress...");
    showNextRating();
  });
  $("rating-slider").addEventListener("input", (e) => {
    $("rating-value").textContent = `当前评分：${e.target.value}`;
  });
  $("rating-next").addEventListener("click", submitRating);
  $("start-experiment").addEventListener("click", startExperiment);
  $("download-data").addEventListener("click", downloadData);
  $("left-btn").addEventListener("click", () => handleChoice("left", "button"));
  $("right-btn").addEventListener("click", () => handleChoice("right", "button"));
  window.addEventListener("keydown", (e) => {
    if (config.responseKeys.left.includes(e.key)) {
      e.preventDefault();
      handleChoice("left", "key");
    }
    if (config.responseKeys.right.includes(e.key)) {
      e.preventDefault();
      handleChoice("right", "key");
    }
  });
  statusLine("Waiting to start.");
  roundLine();
}

document.addEventListener("DOMContentLoaded", setup);
