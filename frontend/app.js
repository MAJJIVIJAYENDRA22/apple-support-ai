// =========================================================
// APPLE SUPPORT AI - FRONTEND
// =========================================================

// IMPORTANT:
// Browser uses 127.0.0.1.
// FastAPI can listen on 0.0.0.0 internally.
const API_URL = "http://127.0.0.1:8001/predict";


// =========================================================
// DOM ELEMENTS
// =========================================================

const messageInput =
    document.getElementById("message");

const sendButton =
    document.getElementById("send");

const chat =
    document.getElementById("chat");


// =========================================================
// BASIC VALIDATION
// =========================================================

if (!messageInput) {
    console.error(
        "Element #message was not found."
    );
}

if (!sendButton) {
    console.error(
        "Element #send was not found."
    );
}

if (!chat) {
    console.error(
        "Element #chat was not found."
    );
}


// =========================================================
// ADD MESSAGE
// =========================================================

function addMessage(
    text,
    type,
    metadata = null
) {

    const message =
        document.createElement("div");

    message.className =
        `message ${type}`;


    const content =
        document.createElement("div");

    content.className =
        "message-content";


    // -----------------------------------------------------
    // Message text
    // -----------------------------------------------------

    content.textContent =
        text;


    message.appendChild(
        content
    );


    // -----------------------------------------------------
    // AI metadata
    // -----------------------------------------------------

    if (
        type === "assistant" &&
        metadata
    ) {

        const info =
            document.createElement("div");

        info.className =
            "ai-info";


        // Intent

        const intent =
            document.createElement("span");

        intent.innerHTML =
            `Intent: <strong>${escapeHTML(
                metadata.intent || "Unknown"
            )}</strong>`;


        info.appendChild(
            intent
        );


        // Confidence

        const confidence =
            document.createElement("span");

        confidence.innerHTML =
            `Confidence: <strong>${escapeHTML(
                metadata.confidence || "N/A"
            )}</strong>`;


        info.appendChild(
            confidence
        );


        // Similarity

        if (
            metadata.similarity !== null &&
            metadata.similarity !== undefined
        ) {

            const similarity =
                document.createElement("span");

            similarity.innerHTML =
                `Similarity: <strong>${escapeHTML(
                    metadata.similarity
                )}</strong>`;


            info.appendChild(
                similarity
            );
        }


        // Status

        if (metadata.status) {

            const status =
                document.createElement("span");

            status.innerHTML =
                `Status: <strong>${escapeHTML(
                    metadata.status
                )}</strong>`;


            info.appendChild(
                status
            );
        }


        content.appendChild(
            info
        );


        // -------------------------------------------------
        // Escalation message
        // -------------------------------------------------

        if (
            metadata.escalation === true
        ) {

            const escalation =
                document.createElement(
                    "div"
                );

            escalation.className =
                "escalation";


            escalation.textContent =
                "This issue requires further support because the AI was not confident enough to provide a reliable answer.";


            content.appendChild(
                escalation
            );
        }
    }


    chat.appendChild(
        message
    );


    scrollChatToBottom();
}


// =========================================================
// ESCAPE HTML
// =========================================================

function escapeHTML(value) {

    const div =
        document.createElement("div");

    div.textContent =
        String(value);

    return div.innerHTML;
}


// =========================================================
// SCROLL CHAT
// =========================================================

function scrollChatToBottom() {

    if (!chat) {
        return;
    }

    chat.scrollTop =
        chat.scrollHeight;
}


// =========================================================
// LOADING MESSAGE
// =========================================================

function addLoading() {

    if (
        document.getElementById(
            "loading"
        )
    ) {

        return;
    }


    const message =
        document.createElement("div");

    message.className =
        "message assistant";

    message.id =
        "loading";


    const content =
        document.createElement("div");

    content.className =
        "message-content";


    content.textContent =
        "Thinking...";


    message.appendChild(
        content
    );


    chat.appendChild(
        message
    );


    scrollChatToBottom();
}


// =========================================================
// REMOVE LOADING
// =========================================================

function removeLoading() {

    const loading =
        document.getElementById(
            "loading"
        );


    if (loading) {

        loading.remove();

    }
}


// =========================================================
// CHECK API CONNECTION
// =========================================================

async function checkAPIConnection() {

    try {

        const response =
            await fetch(
                "http://127.0.0.1:8000/health",
                {
                    method: "GET",
                    headers: {
                        "Accept":
                            "application/json"
                    }
                }
            );


        if (!response.ok) {

            throw new Error(
                `API returned HTTP ${response.status}`
            );
        }


        const data =
            await response.json();


        console.log(
            "Apple Support AI API connected:",
            data
        );


        // -------------------------------------------------
        // Optional connection indicator
        // -------------------------------------------------

        const indicator =
            document.querySelector(
                ".api-status"
            );


        if (indicator) {

            indicator.textContent =
                "● API Connected";

            indicator.classList.add(
                "connected"
            );

            indicator.classList.remove(
                "disconnected"
            );
        }


        return true;


    } catch (error) {

        console.error(
            "API connection failed:",
            error
        );


        const indicator =
            document.querySelector(
                ".api-status"
            );


        if (indicator) {

            indicator.textContent =
                "● API Disconnected";

            indicator.classList.add(
                "disconnected"
            );

            indicator.classList.remove(
                "connected"
            );
        }


        return false;
    }
}


// =========================================================
// SEND MESSAGE
// =========================================================

async function sendMessage() {

    if (
        !messageInput ||
        !sendButton ||
        !chat
    ) {

        return;
    }


    const message =
        messageInput.value.trim();


    // -----------------------------------------------------
    // Empty message
    // -----------------------------------------------------

    if (!message) {

        return;
    }


    // -----------------------------------------------------
    // Disable UI
    // -----------------------------------------------------

    sendButton.disabled =
        true;

    messageInput.disabled =
        true;


    // -----------------------------------------------------
    // Display customer message
    // -----------------------------------------------------

    addMessage(
        message,
        "user"
    );


    messageInput.value =
        "";


    // -----------------------------------------------------
    // Loading
    // -----------------------------------------------------

    addLoading();


    try {

        // =================================================
        // API REQUEST
        // =================================================

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            message:
                                message
                        })
                }
            );


        // =================================================
        // RESPONSE CONTENT TYPE
        // =================================================

        const contentType =
            response.headers.get(
                "content-type"
            );


        let data;


        if (
            contentType &&
            contentType.includes(
                "application/json"
            )
        ) {

            data =
                await response.json();

        } else {

            const text =
                await response.text();


            throw new Error(
                `Server returned an unexpected response: ${text}`
            );
        }


        // Remove loading

        removeLoading();


        // =================================================
        // API ERROR
        // =================================================

        if (!response.ok) {

            throw new Error(
                data.detail ||
                data.message ||
                `API request failed with status ${response.status}.`
            );
        }


        console.log(
            "AI API response:",
            data
        );


        // =================================================
        // RECOMMENDED RESPONSE
        // =================================================

        let responseText =
            data.recommended_response;


        // -------------------------------------------------
        // Escalation / no response
        // -------------------------------------------------

        if (
            !responseText
        ) {

            if (
                data.status ===
                "escalation_required"
            ) {

                responseText =
                    "I'm sorry, but I couldn't confidently determine a reliable answer for this issue. Please contact Apple Support for further assistance.";

            } else {

                responseText =
                    "I'm sorry, but I couldn't find a suitable response for your issue. Please try describing the problem in more detail.";

            }
        }


        // =================================================
        // CONFIDENCE
        // =================================================

        let confidence =
            "N/A";


        if (
            typeof data.confidence ===
            "number"
        ) {

            confidence =
                `${(
                    data.confidence * 100
                ).toFixed(2)}%`;
        }


        // =================================================
        // SIMILARITY
        // =================================================

        let similarity =
            null;


        if (
            typeof data.similarity ===
            "number"
        ) {

            similarity =
                `${(
                    data.similarity * 100
                ).toFixed(2)}%`;
        }


        // =================================================
        // ADD AI RESPONSE
        // =================================================

        addMessage(
            responseText,
            "assistant",
            {
                intent:
                    data.predicted_intent ||
                    "Unknown",

                confidence:
                    confidence,

                similarity:
                    similarity,

                status:
                    data.status ||
                    "unknown",

                escalation:
                    data.escalation?.required ===
                    true
            }
        );


    } catch (error) {

        // -------------------------------------------------
        // Remove loading
        // -------------------------------------------------

        removeLoading();


        console.error(
            "Apple Support AI API Error:",
            error
        );


        let errorMessage;


        // =================================================
        // CONNECTION ERROR
        // =================================================

        if (
            error instanceof TypeError &&
            error.message ===
                "Failed to fetch"
        ) {

            errorMessage =
                "Unable to connect to the AI service. Please make sure the FastAPI server is running on http://127.0.0.1:8000 and CORS is enabled.";

        } else {

            errorMessage =
                `Unable to connect to the AI service. ${error.message}`;
        }


        addMessage(
            errorMessage,
            "assistant"
        );
    }


    // =====================================================
    // RE-ENABLE UI
    // =====================================================

    sendButton.disabled =
        false;

    messageInput.disabled =
        false;

    messageInput.focus();
}


// =========================================================
// ENTER KEY
// =========================================================

if (messageInput) {

    messageInput.addEventListener(
        "keydown",
        function(event) {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                sendMessage();
            }
        }
    );
}


// =========================================================
// SEND BUTTON
// =========================================================

if (sendButton) {

    sendButton.addEventListener(
        "click",
        sendMessage
    );
}


// =========================================================
// INITIALIZE
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    function() {

        console.log(
            "Apple Support AI frontend initialized."
        );

        console.log(
            "API URL:",
            API_URL
        );


        // Check backend connection

        checkAPIConnection();
    }
);