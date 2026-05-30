const resultBox = document.getElementById("resultBox");
const healthBadge = document.getElementById("healthBadge");
const pdfSelect = document.getElementById("pdfSelect");
const chatFeed = document.getElementById("chatFeed");
const chatInput = document.getElementById("chatInput");
const chatForm = document.getElementById("chatForm");
const chatSendBtn = document.getElementById("chatSendBtn");
const chatClearBtn = document.getElementById("chatClearBtn");
const sessionChip = document.getElementById("sessionChip");
const toolCount = document.getElementById("toolCount");
const memoryBtn = document.getElementById("memoryBtn");
const newChatBtn = document.getElementById("newChatBtn");
const conversationList = document.getElementById("conversationList");
const conversationCount = document.getElementById("conversationCount");
const emptyConversations = document.getElementById("emptyConversations");
const codeWorkspacePathInput = document.getElementById("codeWorkspacePath");
const chooseWorkspaceBtn = document.getElementById("chooseWorkspaceBtn");
const codePythonExecutableInput = document.getElementById("codePythonExecutable");
const choosePythonBtn = document.getElementById("choosePythonBtn");
const directoryPickerModal = document.getElementById("directoryPickerModal");
const directoryPickerPath = document.getElementById("directoryPickerPath");
const directoryPickerList = document.getElementById("directoryPickerList");
const openDirectoryPathBtn = document.getElementById("openDirectoryPathBtn");
const selectDirectoryBtn = document.getElementById("selectDirectoryBtn");
const closeDirectoryPickerBtn = document.getElementById("closeDirectoryPickerBtn");
const pythonPickerModal = document.getElementById("pythonPickerModal");
const pythonPickerPath = document.getElementById("pythonPickerPath");
const pythonPickerList = document.getElementById("pythonPickerList");
const openPythonPathBtn = document.getElementById("openPythonPathBtn");
const selectPythonBtn = document.getElementById("selectPythonBtn");
const closePythonPickerBtn = document.getElementById("closePythonPickerBtn");

const SESSION_KEY = "scholarAgentSessionId";
const CONVERSATIONS_KEY = "scholarAgentConversations";
const CODE_WORKSPACE_KEY = "scholarAgentCodeWorkspacePath";
const CODE_WORKSPACE_IS_PROJECT_KEY = "scholarAgentCodeWorkspaceIsProject";
const CODE_PYTHON_EXECUTABLE_KEY = "scholarAgentCodePythonExecutable";
const DEFAULT_CODE_WORKSPACE_PATH = "D:/scholar agent/scholar code";
const DEFAULT_CODE_PYTHON_EXECUTABLE = "D:/";
const INTRO_MESSAGE = "输入研究问题，例如：检索相关论文，并总结可复现的算法路线。";

let chatSessionId = getOrCreateChatSessionId();
let chatHistory = loadStoredChatHistory(chatSessionId);
let openConversationMenuId = "";
let openConversationMenuPosition = null;
const LEGACY_DEFAULT_CODE_WORKSPACE_PATH = "D:/scholar code";
const rawStoredCodeWorkspacePath = localStorage.getItem(CODE_WORKSPACE_KEY) || "";
const storedCodeWorkspacePath =
  rawStoredCodeWorkspacePath === LEGACY_DEFAULT_CODE_WORKSPACE_PATH ? "" : rawStoredCodeWorkspacePath;
const storedCodeWorkspaceProjectFlag = localStorage.getItem(CODE_WORKSPACE_IS_PROJECT_KEY);
let codeWorkspaceIsProject =
  storedCodeWorkspaceProjectFlag === "true" ||
  (storedCodeWorkspaceProjectFlag === null &&
    Boolean(storedCodeWorkspacePath) &&
    storedCodeWorkspacePath !== DEFAULT_CODE_WORKSPACE_PATH);

if (codeWorkspacePathInput) {
  codeWorkspacePathInput.value = storedCodeWorkspacePath || DEFAULT_CODE_WORKSPACE_PATH;
}
if (codePythonExecutableInput) {
  codePythonExecutableInput.value =
    localStorage.getItem(CODE_PYTHON_EXECUTABLE_KEY) || DEFAULT_CODE_PYTHON_EXECUTABLE;
}
if (rawStoredCodeWorkspacePath === LEGACY_DEFAULT_CODE_WORKSPACE_PATH) {
  localStorage.setItem(CODE_WORKSPACE_KEY, DEFAULT_CODE_WORKSPACE_PATH);
}
if (storedCodeWorkspaceProjectFlag === null) {
  localStorage.setItem(CODE_WORKSPACE_IS_PROJECT_KEY, codeWorkspaceIsProject ? "true" : "false");
}

function getOrCreateChatSessionId() {
  const existing = localStorage.getItem(SESSION_KEY);
  if (existing) {
    return existing;
  }
  const sessionId = createChatSessionId();
  localStorage.setItem(SESSION_KEY, sessionId);
  return sessionId;
}

function createChatSessionId() {
  const suffix =
    window.crypto && window.crypto.randomUUID
      ? window.crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `web-${suffix}`;
}

function shortenSessionId(sessionId) {
  if (!sessionId || sessionId.length <= 18) {
    return sessionId || "pending";
  }
  return `${sessionId.slice(0, 10)}...${sessionId.slice(-5)}`;
}

function chatHistoryKey(sessionId) {
  return `scholarAgentChatHistory:${sessionId}`;
}

function loadStoredChatHistory(sessionId) {
  try {
    const parsed = JSON.parse(localStorage.getItem(chatHistoryKey(sessionId)) || "[]");
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter((item) => item && ["user", "assistant"].includes(item.role) && item.content)
      .slice(-40);
  } catch (_) {
    return [];
  }
}

function saveChatHistory() {
  localStorage.setItem(chatHistoryKey(chatSessionId), JSON.stringify(chatHistory.slice(-40)));
}

function loadConversations() {
  try {
    const parsed = JSON.parse(localStorage.getItem(CONVERSATIONS_KEY) || "[]");
    if (!Array.isArray(parsed)) {
      return [];
    }
    return sortConversations(
      parsed
        .filter((item) => item?.id)
        .map((item) => ({
          id: String(item.id),
          title: String(item.title || "新对话"),
          updatedAt: item.updatedAt || "",
          pinned: Boolean(item.pinned),
          pinnedAt: item.pinnedAt || "",
          manualTitle: Boolean(item.manualTitle),
        }))
    );
  } catch (_) {
    return [];
  }
}

function saveConversations(conversations) {
  localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(sortConversations(conversations).slice(0, 30)));
}

function sortConversations(conversations) {
  return [...conversations].sort((a, b) => {
    if (Boolean(a.pinned) !== Boolean(b.pinned)) {
      return a.pinned ? -1 : 1;
    }
    const leftTime = Date.parse(a.pinned ? a.pinnedAt || a.updatedAt : a.updatedAt) || 0;
    const rightTime = Date.parse(b.pinned ? b.pinnedAt || b.updatedAt : b.updatedAt) || 0;
    return rightTime - leftTime;
  });
}

function ensureConversation(sessionId, title = "新对话") {
  const now = new Date().toISOString();
  const conversations = loadConversations();
  if (!conversations.some((item) => item.id === sessionId)) {
    conversations.unshift({ id: sessionId, title, updatedAt: now });
    saveConversations(conversations);
  }
}

function updateConversation(sessionId, updates = {}) {
  const now = new Date().toISOString();
  let conversations = loadConversations();
  const index = conversations.findIndex((item) => item.id === sessionId);
  if (index === -1) {
    conversations.unshift({ id: sessionId, title: "新对话", updatedAt: now, ...updates });
  } else {
    conversations[index] = { ...conversations[index], ...updates, updatedAt: updates.updatedAt || now };
  }
  saveConversations(conversations);
  renderConversationList();
}

function titleFromMessage(message) {
  const normalized = String(message || "").replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "新对话";
  }
  return normalized.length > 22 ? `${normalized.slice(0, 22)}...` : normalized;
}

function autoTitleConversationFromMessage(sessionId, message) {
  const conversation = loadConversations().find((item) => item.id === sessionId);
  if (conversation?.manualTitle) {
    updateConversation(sessionId);
    return;
  }
  const currentTitle = String(conversation?.title || "").trim();
  if (currentTitle && currentTitle !== "新对话") {
    updateConversation(sessionId);
    return;
  }
  updateConversation(sessionId, { title: titleFromMessage(message), manualTitle: false });
}

function formatConversationTime(value) {
  if (!value) {
    return "刚刚";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "刚刚";
  }
  return date.toLocaleString([], {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function updateSessionUi() {
  if (sessionChip) {
    sessionChip.textContent = shortenSessionId(chatSessionId);
    sessionChip.title = chatSessionId;
  }
}

function renderConversationList() {
  if (!conversationList) {
    return;
  }
  const conversations = loadConversations();
  conversationList.innerHTML = "";

  if (conversationCount) {
    conversationCount.textContent = String(conversations.length);
  }
  if (emptyConversations) {
    emptyConversations.classList.toggle("hidden", conversations.length > 0);
  }

  for (const item of conversations) {
    const row = document.createElement("div");
    row.className = `conversation-row${item.id === chatSessionId ? " active" : ""}`;
    if (item.pinned) {
      row.classList.add("pinned");
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "conversation-item";
    button.title = item.id;

    const title = document.createElement("span");
    title.className = "conversation-title";
    title.textContent = item.title || "新对话";

    const meta = document.createElement("span");
    meta.className = "conversation-meta";
    meta.textContent = `${item.pinned ? "置顶 · " : ""}${formatConversationTime(item.updatedAt)}`;

    button.appendChild(title);
    button.appendChild(meta);
    button.addEventListener("click", () => switchConversation(item.id));

    const actionWrap = document.createElement("div");
    actionWrap.className = "conversation-actions";

    const menuButton = document.createElement("button");
    menuButton.type = "button";
    menuButton.className = "conversation-menu-button";
    menuButton.setAttribute("aria-label", `打开对话菜单：${item.title || "新对话"}`);
    menuButton.setAttribute("aria-haspopup", "menu");
    menuButton.setAttribute("aria-expanded", item.id === openConversationMenuId ? "true" : "false");
    menuButton.title = "更多操作";
    menuButton.textContent = "...";
    menuButton.addEventListener("click", (event) => {
      event.stopPropagation();
      if (openConversationMenuId === item.id) {
        closeConversationMenu();
      } else {
        openConversationMenuId = item.id;
        openConversationMenuPosition = getConversationMenuPosition(menuButton);
      }
      renderConversationList();
    });

    const menu = document.createElement("div");
    menu.className = "conversation-menu";
    menu.setAttribute("role", "menu");
    menu.classList.toggle("hidden", item.id !== openConversationMenuId);
    if (item.id === openConversationMenuId && openConversationMenuPosition) {
      menu.style.left = `${openConversationMenuPosition.left}px`;
      menu.style.top = `${openConversationMenuPosition.top}px`;
    }
    menu.appendChild(createConversationMenuItem("重命名", () => renameConversation(item.id)));
    menu.appendChild(createConversationMenuItem(item.pinned ? "取消置顶" : "置顶", () => togglePinConversation(item.id)));
    menu.appendChild(createConversationMenuItem("删除", () => deleteConversation(item.id), "danger"));

    actionWrap.appendChild(menuButton);
    actionWrap.appendChild(menu);
    row.appendChild(button);
    row.appendChild(actionWrap);
    conversationList.appendChild(row);
  }
}

function getConversationMenuPosition(anchor) {
  const rect = anchor.getBoundingClientRect();
  const menuWidth = 112;
  const margin = 8;
  return {
    left: Math.max(margin, Math.min(rect.right - menuWidth, window.innerWidth - menuWidth - margin)),
    top: Math.min(rect.bottom + 4, window.innerHeight - 112),
  };
}

function closeConversationMenu() {
  openConversationMenuId = "";
  openConversationMenuPosition = null;
}

function createConversationMenuItem(label, onClick, extraClass = "") {
  const item = document.createElement("button");
  item.type = "button";
  item.className = `conversation-menu-item${extraClass ? ` ${extraClass}` : ""}`;
  item.setAttribute("role", "menuitem");
  item.textContent = label;
  item.addEventListener("click", (event) => {
    event.stopPropagation();
    closeConversationMenu();
    onClick();
  });
  return item;
}

function renderChatFeed() {
  chatFeed.innerHTML = "";
  if (chatHistory.length === 0) {
    appendBubble("assistant", INTRO_MESSAGE);
    return;
  }
  for (const message of chatHistory) {
    appendBubble(message.role, message.content);
  }
}

function setResult(title, payload) {
  const now = new Date().toLocaleTimeString();
  resultBox.textContent = `[${now}] ${title}\n\n${JSON.stringify(payload, null, 2)}`;
}

function setError(title, errorText) {
  const now = new Date().toLocaleTimeString();
  resultBox.textContent = `[${now}] ${title}\n\n${formatErrorDetail(errorText)}`;
}

function formatErrorDetail(detail) {
  if (detail == null) {
    return "Unknown error";
  }
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        const loc = Array.isArray(item?.loc) ? item.loc.join(".") : "";
        const msg = item?.msg || JSON.stringify(item);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("\n");
  }
  if (detail.message) {
    return String(detail.message);
  }
  try {
    return JSON.stringify(detail, null, 2);
  } catch (_) {
    return String(detail);
  }
}

async function requestJson(url, options = {}) {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      if (data?.detail) {
        detail = formatErrorDetail(data.detail);
      }
    } catch (_) {
      // keep default detail
    }
    throw new Error(detail);
  }

  return resp.json();
}

async function loadHealth() {
  try {
    const health = await requestJson("/api/health");
    if (health.status === "ok") {
      healthBadge.textContent = "API online";
      healthBadge.className = "status-box status-ok";
    } else {
      healthBadge.textContent = "API degraded";
      healthBadge.className = "status-box status-fail";
    }
  } catch (err) {
    healthBadge.textContent = `API error: ${err.message}`;
    healthBadge.className = "status-box status-fail";
  }
}

async function loadTools() {
  try {
    const data = await requestJson("/api/tools");
    const tools = data.tools || [];
    if (toolCount) {
      toolCount.textContent = `${tools.length} ready`;
    }
  } catch (err) {
    if (toolCount) {
      toolCount.textContent = "unavailable";
    }
    setError("Load tools failed", err.message);
  }
}

async function loadPdfFiles() {
  try {
    const data = await requestJson("/api/papers/files");
    const files = data.files || [];

    pdfSelect.innerHTML = "";
    if (files.length === 0) {
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = "No PDF files in workspace/papers";
      pdfSelect.appendChild(empty);
      return;
    }

    for (const file of files) {
      const option = document.createElement("option");
      option.value = file;
      option.textContent = file;
      pdfSelect.appendChild(option);
    }
  } catch (err) {
    setError("Load PDF list failed", err.message);
  }
}

function appendBubble(role, content, extraClass = "") {
  const node = document.createElement("div");
  node.className = `bubble ${role}${extraClass ? ` ${extraClass}` : ""}`;
  if (role === "assistant" && !extraClass.includes("progress")) {
    node.classList.add("markdown-body");
    node.innerHTML = renderMarkdown(content);
  } else {
    node.textContent = content;
  }
  chatFeed.appendChild(node);
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function renderMarkdown(markdown) {
  const source = String(markdown || "").replace(/\r\n/g, "\n");
  const lines = source.split("\n");
  const html = [];
  let paragraph = [];
  let listItems = [];
  let orderedListItems = [];
  let tableLines = [];
  let inCode = false;
  let codeLang = "";
  let codeLines = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (listItems.length) {
      html.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      listItems = [];
    }
    if (orderedListItems.length) {
      html.push(`<ol>${orderedListItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
      orderedListItems = [];
    }
  };
  const flushTable = () => {
    if (tableLines.length) {
      html.push(renderMarkdownTable(tableLines));
      tableLines = [];
    }
  };
  const flushBlocks = () => {
    flushParagraph();
    flushList();
    flushTable();
  };

  for (const line of lines) {
    const fence = line.match(/^```([A-Za-z0-9_-]*)\s*$/);
    if (fence) {
      if (inCode) {
        html.push(
          `<pre><code${codeLang ? ` class="language-${escapeAttribute(codeLang)}"` : ""}>${escapeHtml(codeLines.join("\n"))}</code></pre>`
        );
        inCode = false;
        codeLang = "";
        codeLines = [];
      } else {
        flushBlocks();
        inCode = true;
        codeLang = fence[1] || "";
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (!line.trim()) {
      flushBlocks();
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushBlocks();
      const level = Math.min(heading[1].length, 6);
      html.push(`<h${level}>${renderInlineMarkdown(heading[2].trim())}</h${level}>`);
      continue;
    }

    const list = line.match(/^\s*[-*+]\s+(.+)$/);
    if (list) {
      flushParagraph();
      flushTable();
      if (orderedListItems.length) {
        flushList();
      }
      listItems.push(list[1].trim());
      continue;
    }

    const orderedList = line.match(/^\s*\d+\.\s+(.+)$/);
    if (orderedList) {
      flushParagraph();
      flushTable();
      if (listItems.length) {
        flushList();
      }
      orderedListItems.push(orderedList[1].trim());
      continue;
    }

    if (/^\s*\|.+\|\s*$/.test(line)) {
      flushParagraph();
      flushList();
      tableLines.push(line.trim());
      continue;
    }

    flushList();
    flushTable();
    paragraph.push(line.trim());
  }

  if (inCode) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  flushBlocks();
  return html.join("");
}

function renderMarkdownTable(lines) {
  const rows = lines.map((line) =>
    line
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((cell) => cell.trim())
  );
  if (rows.length === 0) {
    return "";
  }
  const hasDivider = rows.length > 1 && rows[1].every((cell) => /^:?-{3,}:?$/.test(cell));
  const head = rows[0];
  const body = hasDivider ? rows.slice(2) : rows.slice(1);
  return (
    "<div class=\"markdown-table-wrap\"><table><thead><tr>" +
    head.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("") +
    "</tr></thead><tbody>" +
    body
      .map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`)
      .join("") +
    "</tbody></table></div>"
  );
}

function renderInlineMarkdown(text) {
  const tokens = [];
  let safe = escapeHtml(text).replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE${tokens.length}@@`;
    tokens.push(`<code>${code}</code>`);
    return token;
  });
  safe = safe
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  tokens.forEach((value, index) => {
    safe = safe.replace(`@@CODE${index}@@`, value);
  });
  return safe;
}

function addChatMessage(role, content) {
  appendBubble(role, content);
  chatHistory.push({ role, content });
  saveChatHistory();
}

function renderToolTrace(trace) {
  if (!Array.isArray(trace) || trace.length === 0) {
    return;
  }

  const lines = trace.map((item, idx) => {
    const cached = item.cached ? " (cached)" : "";
    return `${idx + 1}. ${item.tool} -> ${item.status}${cached}`;
  });
  appendBubble("assistant", `Tool trace:\n${lines.join("\n")}`, "meta");
}

function setChatPending(isPending) {
  chatSendBtn.disabled = isPending;
  chatInput.disabled = isPending;
  if (codeWorkspacePathInput) {
    codeWorkspacePathInput.disabled = isPending;
  }
  if (chooseWorkspaceBtn) {
    chooseWorkspaceBtn.disabled = isPending;
  }
  if (codePythonExecutableInput) {
    codePythonExecutableInput.disabled = isPending;
  }
  if (choosePythonBtn) {
    choosePythonBtn.disabled = isPending;
  }
}

function bindChatForm() {
  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message) {
      return;
    }

    addChatMessage("user", message);
    autoTitleConversationFromMessage(chatSessionId, message);
    const codeWorkspacePath = (codeWorkspacePathInput?.value || DEFAULT_CODE_WORKSPACE_PATH).trim();
    const codePythonExecutable = (codePythonExecutableInput?.value || DEFAULT_CODE_PYTHON_EXECUTABLE).trim();
    localStorage.setItem(CODE_WORKSPACE_KEY, codeWorkspacePath || DEFAULT_CODE_WORKSPACE_PATH);
    localStorage.setItem(CODE_PYTHON_EXECUTABLE_KEY, codePythonExecutable || DEFAULT_CODE_PYTHON_EXECUTABLE);
    chatInput.value = "";
    setChatPending(true);

    // Create a progress bubble that will update in real-time
    const progressBubble = document.createElement("div");
    progressBubble.className = "bubble assistant progress";
    renderProgressBubble(progressBubble, {
      title: "Supervisor 正在规划...",
      detail: "正在分析用户问题并生成模块计划。",
      state: "running",
    });
    chatFeed.appendChild(progressBubble);
    chatFeed.scrollTop = chatFeed.scrollHeight;

    const payload = {
      message,
      history: chatHistory
        .slice(0, -1)
        .slice(-8)
        .map((item) => ({
          role: item.role,
          content: String(item.content || "").slice(0, 6000),
        })),
      session_id: chatSessionId,
      code_workspace_path: codeWorkspacePath || DEFAULT_CODE_WORKSPACE_PATH,
      code_workspace_is_project: codeWorkspaceIsProject,
      code_python_executable: codePythonExecutable || DEFAULT_CODE_PYTHON_EXECUTABLE,
      temperature: 0.2,
      max_steps: 12,
    };

    try {
      const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        let detail = `HTTP ${resp.status}`;
        try {
          const errData = await resp.json();
          if (errData?.detail) detail = formatErrorDetail(errData.detail);
        } catch (_) {}
        throw new Error(detail);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalAnswer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const parsed = parseSseEvent(rawEvent);
          if (parsed) {
            handleStreamEvent(parsed.event, parsed.data, progressBubble);
            if (parsed.event === "done") {
              finalAnswer = parsed.data.answer || "";
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }

      // Remove progress bubble
      if (progressBubble.parentNode) {
        progressBubble.remove();
      }

      if (finalAnswer) {
        addChatMessage("assistant", finalAnswer);
        updateConversation(chatSessionId);
        setResult("Chat", { status: "success", answer: finalAnswer });
      } else {
        const errText = "No answer received from stream.";
        addChatMessage("assistant", errText);
        updateConversation(chatSessionId);
      }
    } catch (err) {
      if (progressBubble.parentNode) {
        progressBubble.remove();
      }
      const errText = `Request failed: ${err.message}`;
      addChatMessage("assistant", errText);
      updateConversation(chatSessionId);
      setError("Chat", err.message);
    } finally {
      setChatPending(false);
      chatInput.focus();
    }
  });
}

function parseSseEvent(rawEvent) {
  const lines = rawEvent.replace(/\r\n/g, "\n").split("\n");
  let event = "";
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (!event || dataLines.length === 0) {
    return null;
  }

  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch (_) {
    return null;
  }
}

function handleStreamEvent(type, data, progressBubble) {
  switch (type) {
    case "status":
      renderProgressBubble(progressBubble, {
        title: "工作流已启动",
        detail: `Session ${data.session_id || ""}`,
        state: "running",
      });
      break;

    case "supervisor":
      // Supervisor made a routing decision
      if (data.decision === "FINISH") {
        renderProgressBubble(progressBubble, {
          title: data.label,
          detail: "专家模块已完成，正在整合结果。",
          state: "running",
        });
      } else {
        renderProgressBubble(progressBubble, {
          title: `Supervisor → ${data.label}`,
          detail: data.task || data.reason || "",
          state: "running",
        });
      }
      chatFeed.scrollTop = chatFeed.scrollHeight;
      break;

    case "progress":
      renderProgressBubble(progressBubble, {
        title: data.title || data.label || "Agent 状态",
        detail: data.detail || "",
        state: data.status === "fail" ? "error" : "running",
      });
      chatFeed.scrollTop = chatFeed.scrollHeight;
      break;

    case "expert_output":
      // An expert finished its analysis
      renderProgressBubble(progressBubble, {
        title: `${data.label} 完成`,
        detail: data.summary ? `${data.summary}…` : "",
        state: "running",
      });
      chatFeed.scrollTop = chatFeed.scrollHeight;
      break;

    case "done":
      renderProgressBubble(progressBubble, {
        title: "整合完成",
        detail: `共执行 ${data.steps || 0} 个图节点。`,
        state: "done",
      });
      chatFeed.scrollTop = chatFeed.scrollHeight;
      break;

    case "error":
      renderProgressBubble(progressBubble, {
        title: "出错",
        detail: data.message || "",
        state: "error",
      });
      break;
  }
}

function renderProgressBubble(node, current) {
  const dotClass =
    current.state === "done" ? " done" : current.state === "error" ? " error" : "";

  node.innerHTML =
    '<div class="progress-current"><span class="progress-dot' +
    dotClass +
    '"></span><strong>' +
    escapeHtml(current.title || "Agent 正在处理...") +
    "</strong>" +
    (current.detail ? "<small>" + escapeHtml(current.detail) + "</small>" : "") +
    "</div>";
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function escapeAttribute(text) {
  return escapeHtml(text).replace(/"/g, "&quot;");
}

function bindChatClearButton() {
  chatClearBtn.addEventListener("click", () => {
    chatHistory = [];
    saveChatHistory();
    chatFeed.innerHTML = "";
    appendBubble("assistant", "对话已清空。输入新的研究问题后，Supervisor 会重新规划模块调用。");
    updateConversation(chatSessionId, { title: "新对话" });
    setResult("Chat", { status: "cleared" });

    fetch(`/api/chat/memory/${encodeURIComponent(chatSessionId)}`, {
      method: "DELETE",
    }).catch(() => {
      // Local UI clear should still succeed if the server is unavailable.
    });
  });
}

function switchConversation(sessionId) {
  if (!sessionId || sessionId === chatSessionId) {
    return;
  }
  chatSessionId = sessionId;
  localStorage.setItem(SESSION_KEY, chatSessionId);
  chatHistory = loadStoredChatHistory(chatSessionId);
  ensureConversation(chatSessionId);
  updateSessionUi();
  renderConversationList();
  renderChatFeed();
  setResult("Conversation", { status: "switched", session_id: chatSessionId });
  chatInput.focus();
}

function renameConversation(sessionId) {
  const conversations = loadConversations();
  const current = conversations.find((item) => item.id === sessionId);
  const nextTitle = window.prompt("重命名对话", current?.title || "新对话");
  if (nextTitle == null) {
    renderConversationList();
    return;
  }
  const normalized = nextTitle.replace(/\s+/g, " ").trim();
  if (!normalized) {
    renderConversationList();
    return;
  }
  updateConversation(sessionId, {
    title: normalized.length > 40 ? `${normalized.slice(0, 40)}...` : normalized,
    manualTitle: true,
  });
  setResult("Conversation", { status: "renamed", session_id: sessionId });
}

function togglePinConversation(sessionId) {
  const conversations = loadConversations();
  const index = conversations.findIndex((item) => item.id === sessionId);
  if (index === -1) {
    return;
  }
  const nextPinned = !conversations[index].pinned;
  conversations[index] = {
    ...conversations[index],
    pinned: nextPinned,
    pinnedAt: nextPinned ? new Date().toISOString() : "",
  };
  saveConversations(conversations);
  renderConversationList();
  setResult("Conversation", { status: nextPinned ? "pinned" : "unpinned", session_id: sessionId });
}

function deleteConversation(sessionId) {
  if (!sessionId) {
    return;
  }

  const conversations = loadConversations().filter((item) => item.id !== sessionId);
  saveConversations(conversations);
  localStorage.removeItem(chatHistoryKey(sessionId));

  fetch(`/api/chat/memory/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  }).catch(() => {
    // Local deletion should still succeed if the server is unavailable.
  });

  if (sessionId === chatSessionId) {
    const nextConversation = conversations[0];
    if (nextConversation) {
      chatSessionId = nextConversation.id;
      localStorage.setItem(SESSION_KEY, chatSessionId);
      chatHistory = loadStoredChatHistory(chatSessionId);
    } else {
      chatSessionId = createChatSessionId();
      localStorage.setItem(SESSION_KEY, chatSessionId);
      chatHistory = [];
      saveChatHistory();
      ensureConversation(chatSessionId);
    }
    updateSessionUi();
    renderChatFeed();
  }

  renderConversationList();
  setResult("Conversation", { status: "deleted", session_id: sessionId });
  chatInput.focus();
}

function bindConversationMenuDismissal() {
  document.addEventListener("click", () => {
    if (!openConversationMenuId) {
      return;
    }
    closeConversationMenu();
    renderConversationList();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !openConversationMenuId) {
      return;
    }
    closeConversationMenu();
    renderConversationList();
  });
  window.addEventListener("resize", () => {
    if (!openConversationMenuId) {
      return;
    }
    closeConversationMenu();
    renderConversationList();
  });
  conversationList?.addEventListener("scroll", () => {
    if (!openConversationMenuId) {
      return;
    }
    closeConversationMenu();
    renderConversationList();
  });
}

function startNewConversation() {
  chatSessionId = createChatSessionId();
  localStorage.setItem(SESSION_KEY, chatSessionId);
  chatHistory = [];
  saveChatHistory();
  ensureConversation(chatSessionId);
  updateSessionUi();
  renderConversationList();
  renderChatFeed();
  setResult("Conversation", { status: "created", session_id: chatSessionId });
  chatInput.focus();
}

function bindEnterSubmit(textareaId, formId) {
  const textarea = document.getElementById(textareaId);
  const form = document.getElementById(formId);
  if (!textarea || !form) {
    return;
  }
  textarea.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) {
      return;
    }
    event.preventDefault();
    if (typeof form.requestSubmit === "function") {
      form.requestSubmit();
    } else {
      form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    }
  });
}

function bindNewChatButton() {
  if (!newChatBtn) {
    return;
  }
  newChatBtn.addEventListener("click", startNewConversation);
}

function bindWorkspacePicker() {
  if (!chooseWorkspaceBtn || !directoryPickerModal || !codeWorkspacePathInput) {
    return;
  }
  chooseWorkspaceBtn.addEventListener("click", async () => {
    directoryPickerModal.classList.remove("hidden");
    const current = codeWorkspacePathInput.value.trim();
    if (current) {
      directoryPickerPath.value = current;
      await loadDirectoryPickerPath(current);
    } else {
      await loadDirectoryRoots();
    }
  });

  closeDirectoryPickerBtn?.addEventListener("click", closeDirectoryPicker);
  selectDirectoryBtn?.addEventListener("click", () => {
    const selected = directoryPickerPath?.value.trim();
    if (!selected) {
      return;
    }
    codeWorkspacePathInput.value = selected;
    codeWorkspaceIsProject = true;
    localStorage.setItem(CODE_WORKSPACE_KEY, selected);
    localStorage.setItem(CODE_WORKSPACE_IS_PROJECT_KEY, "true");
    closeDirectoryPicker();
  });
  codeWorkspacePathInput.addEventListener("input", () => {
    codeWorkspaceIsProject = false;
    localStorage.setItem(CODE_WORKSPACE_IS_PROJECT_KEY, "false");
  });
  openDirectoryPathBtn?.addEventListener("click", async () => {
    const selected = directoryPickerPath?.value.trim();
    if (selected) {
      await loadDirectoryPickerPath(selected);
    }
  });
  directoryPickerPath?.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    await loadDirectoryPickerPath(directoryPickerPath.value.trim());
  });
}

function bindPythonPicker() {
  if (!choosePythonBtn || !pythonPickerModal || !codePythonExecutableInput) {
    return;
  }
  choosePythonBtn.addEventListener("click", async () => {
    pythonPickerModal.classList.remove("hidden");
    const current = codePythonExecutableInput.value.trim() || DEFAULT_CODE_PYTHON_EXECUTABLE;
    pythonPickerPath.value = current;
    await loadPythonPickerPath(current);
  });

  closePythonPickerBtn?.addEventListener("click", closePythonPicker);
  selectPythonBtn?.addEventListener("click", () => {
    const selected = pythonPickerPath?.value.trim();
    if (!selected) {
      return;
    }
    codePythonExecutableInput.value = selected;
    localStorage.setItem(CODE_PYTHON_EXECUTABLE_KEY, selected);
    closePythonPicker();
  });
  codePythonExecutableInput.addEventListener("input", () => {
    localStorage.setItem(
      CODE_PYTHON_EXECUTABLE_KEY,
      codePythonExecutableInput.value.trim() || DEFAULT_CODE_PYTHON_EXECUTABLE
    );
  });
  openPythonPathBtn?.addEventListener("click", async () => {
    const selected = pythonPickerPath?.value.trim();
    if (selected) {
      await loadPythonPickerPath(selected);
    }
  });
  pythonPickerPath?.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    await loadPythonPickerPath(pythonPickerPath.value.trim());
  });
}

function closeDirectoryPicker() {
  directoryPickerModal?.classList.add("hidden");
}

function closePythonPicker() {
  pythonPickerModal?.classList.add("hidden");
}

async function loadDirectoryRoots() {
  try {
    const data = await requestJson("/api/filesystem/roots");
    renderDirectoryEntries(data.roots || [], "");
  } catch (err) {
    setError("Directory Picker", err.message);
  }
}

async function loadDirectoryPickerPath(path) {
  try {
    const data = await requestJson(`/api/filesystem/directories?path=${encodeURIComponent(path)}`);
    if (directoryPickerPath) {
      directoryPickerPath.value = data.path || path;
    }
    const entries = [];
    if (data.parent) {
      entries.push({ name: "..", path: data.parent });
    }
    entries.push(...(data.directories || []));
    renderDirectoryEntries(entries, data.path || path);
  } catch (err) {
    await loadDirectoryRoots();
    setError("Directory Picker", err.message);
  }
}

async function loadPythonPickerPath(path) {
  try {
    const data = await requestJson(`/api/filesystem/python-files?path=${encodeURIComponent(path)}`);
    if (pythonPickerPath) {
      pythonPickerPath.value = data.selected_file || data.path || path;
    }
    const entries = [];
    if (data.parent) {
      entries.push({ name: "..", path: data.parent, is_dir: true });
    }
    entries.push(...(data.entries || []));
    renderPythonEntries(entries, data.selected_file || data.path || path);
  } catch (err) {
    await loadDirectoryRootsForPython();
    setError("Python Picker", err.message);
  }
}

async function loadDirectoryRootsForPython() {
  try {
    const data = await requestJson("/api/filesystem/roots");
    renderPythonEntries(
      (data.roots || []).map((item) => ({ ...item, is_dir: true })),
      ""
    );
  } catch (err) {
    setError("Python Picker", err.message);
  }
}

function renderPythonEntries(entries, currentPath) {
  if (!pythonPickerList) {
    return;
  }
  pythonPickerList.innerHTML = "";
  if (pythonPickerPath && currentPath) {
    pythonPickerPath.value = currentPath;
  }
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "没有可显示的目录或 Python 解释器";
    pythonPickerList.appendChild(empty);
    return;
  }
  for (const entry of entries) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `directory-item${entry.is_dir ? "" : " file"}`;
    button.textContent = entry.is_dir ? entry.name : `${entry.name}  ·  选择解释器`;
    button.title = entry.path;
    button.addEventListener("click", async () => {
      if (pythonPickerPath) {
        pythonPickerPath.value = entry.path;
      }
      if (entry.is_dir) {
        await loadPythonPickerPath(entry.path);
      }
    });
    pythonPickerList.appendChild(button);
  }
}

function renderDirectoryEntries(entries, currentPath) {
  if (!directoryPickerList) {
    return;
  }
  directoryPickerList.innerHTML = "";
  if (directoryPickerPath && currentPath) {
    directoryPickerPath.value = currentPath;
  }
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "没有可显示的子目录";
    directoryPickerList.appendChild(empty);
    return;
  }
  for (const entry of entries) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "directory-item";
    button.textContent = entry.name;
    button.title = entry.path;
    button.addEventListener("click", async () => {
      if (directoryPickerPath) {
        directoryPickerPath.value = entry.path;
      }
      await loadDirectoryPickerPath(entry.path);
    });
    directoryPickerList.appendChild(button);
  }
}

function bindMemoryButton() {
  if (!memoryBtn) {
    return;
  }
  memoryBtn.addEventListener("click", async () => {
    try {
      const memory = await requestJson(`/api/chat/memory/${encodeURIComponent(chatSessionId)}`);
      setResult("Session Memory", memory);
    } catch (err) {
      setError("Session Memory", err.message);
    }
  });
}

function bindArxivForm() {
  const form = document.getElementById("arxivForm");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = {
      query: document.getElementById("arxivQuery").value.trim(),
      max_results: Number(document.getElementById("arxivMax").value || 3),
      sort_by: document.getElementById("arxivSortBy").value,
      sort_order: document.getElementById("arxivSortOrder").value,
    };

    if (!payload.query) {
      setError("arXiv Search", "query is required");
      return;
    }

    try {
      const result = await requestJson("/api/arxiv/search", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setResult("arXiv Search", result);
    } catch (err) {
      setError("arXiv Search", err.message);
    }
  });
}

function bindLocalSearchForm() {
  const form = document.getElementById("localSearchForm");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = {
      query: document.getElementById("localQuery").value.trim(),
      top_k: Number(document.getElementById("localTopK").value || 3),
    };

    if (!payload.query) {
      setError("Local Search", "query is required");
      return;
    }

    try {
      const result = await requestJson("/api/local/search", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setResult("Local Search", result);
    } catch (err) {
      setError("Local Search", err.message);
    }
  });
}

function bindListDatabaseButton() {
  const button = document.getElementById("listDbBtn");
  button.addEventListener("click", async () => {
    try {
      const result = await requestJson("/api/local/database");
      setResult("List Local Database", result);
    } catch (err) {
      setError("List Local Database", err.message);
    }
  });
}

function bindImportButton() {
  const button = document.getElementById("importBtn");
  button.addEventListener("click", async () => {
    const filename = pdfSelect.value;
    if (!filename) {
      setError("Import PDF", "please select a pdf file first");
      return;
    }

    try {
      const result = await requestJson("/api/papers/import", {
        method: "POST",
        body: JSON.stringify({ filename }),
      });
      setResult("Import PDF", result);
    } catch (err) {
      setError("Import PDF", err.message);
    }
  });
}

function bindRefreshFilesButton() {
  const button = document.getElementById("refreshFilesBtn");
  button.addEventListener("click", async () => {
    await loadPdfFiles();
    setResult("Refresh PDF List", { status: "ok" });
  });
}

async function bootstrap() {
  ensureConversation(chatSessionId);
  updateSessionUi();
  renderConversationList();
  renderChatFeed();

  bindChatForm();
  bindEnterSubmit("chatInput", "chatForm");
  bindEnterSubmit("arxivQuery", "arxivForm");
  bindEnterSubmit("localQuery", "localSearchForm");
  bindChatClearButton();
  bindNewChatButton();
  bindWorkspacePicker();
  bindPythonPicker();
  bindConversationMenuDismissal();
  bindMemoryButton();
  bindArxivForm();
  bindLocalSearchForm();
  bindListDatabaseButton();
  bindImportButton();
  bindRefreshFilesButton();

  await loadHealth();
  await loadTools();
  await loadPdfFiles();
}

bootstrap();
