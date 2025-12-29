const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');

// Add message to chat
function addMessage(content, isUser = false, isHistory = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Add play button for bot messages
    if (!isUser) {
        const playBtn = document.createElement('button');
        playBtn.className = 'btn-icon';
        playBtn.innerHTML = '▶️';
        playBtn.title = 'Read aloud';
        playBtn.style.cssText = 'position: absolute; top: 0.5rem; right: 0.5rem; font-size: 1rem;';
        playBtn.onclick = () => speakText(content, playBtn);
        messageDiv.style.position = 'relative';
        messageDiv.appendChild(playBtn);
    }
    
    // Format content with better styling
    let formattedContent = content;
    if (!isUser) {
        // Convert markdown-style headers
        formattedContent = formattedContent.replace(/## (.*?)\n/g, '<h3 style="margin: 1rem 0 0.5rem 0; color: #2c3e50;">$1</h3>');
        
        // Convert bullet points
        formattedContent = formattedContent.replace(/- (.*?)\n/g, '<li style="margin: 0.3rem 0;">$1</li>');
        formattedContent = formattedContent.replace(/(<li.*?<\/li>)+/g, '<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">$&</ul>');
        
        // Convert numbered lists
        formattedContent = formattedContent.replace(/\d+\. (.*?)\n/g, '<li style="margin: 0.3rem 0;">$1</li>');
        
        // Add paragraph breaks
        formattedContent = formattedContent.replace(/\n\n/g, '</p><p style="margin: 0.8rem 0;">');
        
        // Wrap in paragraph
        formattedContent = `<p style="margin: 0.8rem 0;">${formattedContent}</p>`;
    } else {
        formattedContent = `<p>${formattedContent}</p>`;
    }
    
    contentDiv.innerHTML = formattedContent;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    if (!isHistory) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// Text-to-speech function
function speakText(text, button) {
    // Stop any ongoing speech
    window.speechSynthesis.cancel();
    
    // Clean text for speech
    const cleanText = text.replace(/<[^>]*>/g, '').replace(/[*#]/g, '');
    
    if (button.innerHTML === '⏸️') {
        window.speechSynthesis.cancel();
        button.innerHTML = '▶️';
        return;
    }
    
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    
    utterance.onstart = () => {
        button.innerHTML = '⏸️';
    };
    
    utterance.onend = () => {
        button.innerHTML = '▶️';
    };
    
    utterance.onerror = () => {
        button.innerHTML = '▶️';
    };
    
    window.speechSynthesis.speak(utterance);
}


// Add external resources
function addExternalResources(resources) {
    if (!resources) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    let html = '<div class="external-links">';
    
    // Articles
    if (resources.pdfs && resources.pdfs.length > 0) {
        html += '<h4>📚 Recommended Articles:</h4>';
        resources.pdfs.forEach(pdf => {
            html += `<a href="${pdf.url}" target="_blank">${pdf.title}</a>`;
        });
    }
    
    // Videos
    if (resources.videos && resources.videos.length > 0) {
        html += '<h4>🎥 Recommended Videos:</h4>';
        resources.videos.forEach(video => {
            html += `<a href="${video.url}" target="_blank">${video.title} - ${video.channel}</a>`;
        });
    }
    
    html += '</div>';
    contentDiv.innerHTML = html;
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Image Gallery with navigation and download
let currentGallery = [];
let currentImageIndex = 0;

function addImageGallery(images) {
    currentGallery = images;
    currentImageIndex = 0;
    
    const galleryDiv = document.createElement('div');
    galleryDiv.className = 'message bot-message';
    
    const galleryId = 'gallery_' + Date.now();
    
    galleryDiv.innerHTML = `
        <div class="message-content">
            <div style="position: relative; display: inline-block;">
                <img id="${galleryId}" src="${images[0].url}" alt="${images[0].alt}" 
                     style="width: 400px; height: 300px; object-fit: cover; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); cursor: pointer;">
                <button class="gallery-prev-btn" style="position: absolute; left: 10px; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,0.5); color: white; border: none; padding: 10px 15px; border-radius: 50%; cursor: pointer; font-size: 18px;">❮</button>
                <button class="gallery-next-btn" style="position: absolute; right: 10px; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,0.5); color: white; border: none; padding: 10px 15px; border-radius: 50%; cursor: pointer; font-size: 18px;">❯</button>
                <div style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.7); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                    <span class="gallery-counter">1 / ${images.length}</span>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <button class="gallery-download-btn" class="btn btn-primary" style="padding: 0.5rem 1rem; font-size: 0.9rem; background: #6366f1; color: white; border: none; border-radius: 8px; cursor: pointer;">⬇️ Download</button>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(galleryDiv);
    
    // Add event listeners
    const img = galleryDiv.querySelector(`#${galleryId}`);
    const prevBtn = galleryDiv.querySelector('.gallery-prev-btn');
    const nextBtn = galleryDiv.querySelector('.gallery-next-btn');
    const downloadBtn = galleryDiv.querySelector('.gallery-download-btn');
    const counter = galleryDiv.querySelector('.gallery-counter');
    
    img.onclick = () => openImageModal(images[currentImageIndex].url);
    prevBtn.onclick = () => {
        currentImageIndex = (currentImageIndex - 1 + images.length) % images.length;
        img.src = images[currentImageIndex].url;
        img.alt = images[currentImageIndex].alt;
        counter.textContent = `${currentImageIndex + 1} / ${images.length}`;
    };
    nextBtn.onclick = () => {
        currentImageIndex = (currentImageIndex + 1) % images.length;
        img.src = images[currentImageIndex].url;
        img.alt = images[currentImageIndex].alt;
        counter.textContent = `${currentImageIndex + 1} / ${images.length}`;
    };
    downloadBtn.onclick = () => {
        const link = document.createElement('a');
        link.href = images[currentImageIndex].url;
        link.download = `image_${currentImageIndex + 1}.jpg`;
        link.target = '_blank';
        link.click();
    };
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}



function openImageModal(imageUrl) {
    const modal = document.createElement('div');
    modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); display: flex; align-items: center; justify-content: center; z-index: 10000;';
    modal.innerHTML = `
        <img src="${imageUrl}" style="max-width: 90%; max-height: 90%; border-radius: 8px;">
        <button onclick="this.parentElement.remove()" style="position: absolute; top: 20px; right: 20px; background: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 18px;">✕</button>
    `;
    modal.onclick = (e) => { if (e.target === modal) modal.remove(); };
    document.body.appendChild(modal);
}



// Send message
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Add user message
    addMessage(message, true);
    messageInput.value = '';
    
    // Show loading
    addMessage('Thinking...', false);
    
    try {
        // Fetch images in parallel with RAG response
        const [ragResponse, imageResponse] = await Promise.all([
            fetch('/api/chat/message', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message})
            }),
            fetch('/api/search/web', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: message})
            }).catch(() => ({json: () => ({images: []})}))
        ]);
        
        const data = await ragResponse.json();
        const imageData = await imageResponse.json();
        
        // Debug logging
        console.log('[DEBUG] Response data:', data);
        console.log('[DEBUG] is_new_topic:', data.is_new_topic);
        console.log('[DEBUG] topic:', data.topic);
        
        // Remove loading message
        chatMessages.removeChild(chatMessages.lastChild);
        
        // Add images if available
        if (imageData.images && imageData.images.length > 0) {
            addImageGallery(imageData.images);
            
            // Save images with the response
            const imagesJson = JSON.stringify(imageData.images);
            await fetch('/api/chat/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    query: `[IMAGES]${message}`,
                    response: imagesJson
                })
            });
        }
        
        // Add bot response
        const botMessageDiv = chatMessages.lastChild;
        addMessage(data.response, false);
        
        // Add action buttons if topic exists
        if (data.topic) {
            addMessageActions(chatMessages.lastChild, data.topic);
        }
        
        // Add external resources
        if (data.external_resources) {
            addExternalResources(data.external_resources);
        }
        
        // Offer quiz for new topics in chat
        console.log('[DEBUG] Checking quiz offer condition...');
        console.log('[DEBUG] is_new_topic:', data.is_new_topic);
        console.log('[DEBUG] topic:', data.topic);
        
        if (data.is_new_topic === true && data.topic) {
            console.log('[DEBUG] Showing quiz offer for topic:', data.topic);
            const quizPrompt = document.createElement('div');
            quizPrompt.className = 'message bot-message quiz-offer';
            quizPrompt.innerHTML = `
                <div class="message-content">
                    <p style="font-weight: bold; color: #6366f1;">🎯 This is a new topic for you!</p>
                    <p>Would you like to take a quick quiz to test your understanding?</p>
                    <div style="margin-top: 1rem; display: flex; gap: 0.5rem;">
                        <button onclick="takeQuickQuiz('${data.topic}')" class="btn btn-primary">✅ Yes, Take Quiz</button>
                        <button onclick="this.closest('.quiz-offer').remove()" class="btn btn-secondary">❌ No, Thanks</button>
                    </div>
                </div>
            `;
            chatMessages.appendChild(quizPrompt);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } else {
            console.log('[DEBUG] Quiz offer NOT shown. Reason:', !data.is_new_topic ? 'Not new topic' : 'No topic provided');
        }
        
    } catch (error) {
        chatMessages.removeChild(chatMessages.lastChild);
        addMessage('Sorry, something went wrong. Please try again.', false);
        console.error('Chat error:', error);
    }
});

// Load chat history
async function loadChatHistory() {
    try {
        const response = await fetch('/api/chat/history');
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            data.history.forEach(item => {
                // Check if this is an image entry
                if (item.query.startsWith('[IMAGES]')) {
                    try {
                        const images = JSON.parse(item.response);
                        if (Array.isArray(images) && images.length > 0) {
                            currentGallery = images;
                            currentImageIndex = 0;
                            addImageGallery(images);
                        }
                    } catch (e) {
                        console.error('Failed to parse images:', e);
                    }
                } else {
                    addMessage(item.query, true, true);
                    addMessage(item.response, false, true);
                }
            });
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    } catch (error) {
        console.error('Failed to load chat history:', error);
    }
}

// Take quick quiz for new topic
async function takeQuickQuiz(topic) {
    // Remove quiz offer
    const quizOffer = document.querySelector('.quiz-offer');
    if (quizOffer) quizOffer.remove();
    
    // Show loading
    addMessage('Generating quiz questions...', false);
    
    try {
        const response = await fetch('/api/quiz/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                topic: topic,
                difficulty: 'easy',
                num_questions: 3
            })
        });
        
        const data = await response.json();
        
        // Remove loading
        chatMessages.removeChild(chatMessages.lastChild);
        
        if (data.questions && data.questions.length > 0) {
            displayInlineQuiz(data.questions, topic);
        } else {
            addMessage('Sorry, could not generate quiz questions. Please try again.', false);
        }
    } catch (error) {
        chatMessages.removeChild(chatMessages.lastChild);
        addMessage('Error generating quiz. Please try again.', false);
        console.error('Quiz generation error:', error);
    }
}

// Display quiz inline in chat
function displayInlineQuiz(questions, topic) {
    const quizDiv = document.createElement('div');
    quizDiv.className = 'message bot-message inline-quiz';
    
    let html = `
        <div class="message-content">
            <h3>📝 Quick Assessment: ${topic}</h3>
            <form id="inlineQuizForm">
    `;
    
    questions.forEach((q, index) => {
        html += `
            <div class="quiz-question" style="margin: 1rem 0; padding: 1rem; background: #f5f5f5; border-radius: 8px;">
                <p style="font-weight: bold; margin-bottom: 0.5rem;">Q${index + 1}: ${q.question}</p>
                <div class="quiz-options">
        `;
        
        if (q.options && q.options.length > 0) {
            q.options.forEach(option => {
                const value = option.charAt(0);
                html += `
                    <label style="display: block; margin: 0.3rem 0; cursor: pointer;">
                        <input type="radio" name="q${index}" value="${value}" required>
                        ${option}
                    </label>
                `;
            });
        }
        
        html += `
                </div>
            </div>
        `;
    });
    
    html += `
                <button type="submit" class="btn btn-primary" style="margin-top: 1rem;">Submit Quiz</button>
            </form>
        </div>
    `;
    
    quizDiv.innerHTML = html;
    chatMessages.appendChild(quizDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Handle quiz submission
    const form = document.getElementById('inlineQuizForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const answers = [];
        questions.forEach((q, index) => {
            const selected = form.querySelector(`input[name="q${index}"]:checked`);
            answers.push(selected ? selected.value : '');
        });
        
        // Submit quiz
        try {
            const response = await fetch('/api/quiz/submit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    topic: topic,
                    difficulty: 'easy',
                    answers: answers,
                    questions: questions
                })
            });
            
            const result = await response.json();
            
            console.log('[CHAT QUIZ] Quiz submitted successfully:', result);
            
            // Show results
            const resultDiv = document.createElement('div');
            resultDiv.className = 'message bot-message';
            resultDiv.innerHTML = `
                <div class="message-content">
                    <h3>✅ Quiz Results</h3>
                    <p><strong>Score:</strong> ${result.score}/${result.total} (${result.percentage.toFixed(1)}%)</p>
                    <p><strong>Level:</strong> ${result.level.toUpperCase()}</p>
                    <p>${result.percentage >= 70 ? '🎉 Great job! You have good understanding!' : '📚 Keep learning! Review the material and try again.'}</p>
                    <p style="margin-top: 1rem; color: #6366f1; font-size: 0.9rem;">💾 Quiz saved to history - view it in the Quiz page!</p>
                </div>
            `;
            chatMessages.appendChild(resultDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Remove quiz form
            quizDiv.remove();
            
        } catch (error) {
            addMessage('Error submitting quiz. Please try again.', false);
            console.error('Quiz submission error:', error);
        }
    });
}

// Export chat functionality
document.getElementById('exportChatBtn').addEventListener('click', async () => {
    try {
        const response = await fetch('/api/chat/export');
        const data = await response.json();
        
        if (data.content) {
            const blob = new Blob([data.content], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chat_history_${new Date().toISOString().split('T')[0]}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        }
    } catch (error) {
        console.error('Export error:', error);
        alert('Failed to export chat history');
    }
});

// Add favorite and review buttons to bot messages
function addMessageActions(messageDiv, topic) {
    if (!topic) return;
    
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'message-actions';
    actionsDiv.style.cssText = 'margin-top: 0.5rem; display: flex; gap: 0.5rem;';
    
    const favoriteBtn = document.createElement('button');
    favoriteBtn.className = 'btn-icon';
    favoriteBtn.innerHTML = '⭐';
    favoriteBtn.title = 'Mark as favorite';
    favoriteBtn.onclick = () => toggleFavorite(topic, favoriteBtn);
    
    const reviewBtn = document.createElement('button');
    reviewBtn.className = 'btn-icon';
    reviewBtn.innerHTML = '🔄';
    reviewBtn.title = 'Review this topic';
    reviewBtn.onclick = () => reviewTopic(topic);
    
    actionsDiv.appendChild(favoriteBtn);
    actionsDiv.appendChild(reviewBtn);
    messageDiv.querySelector('.message-content').appendChild(actionsDiv);
}

async function toggleFavorite(topic, btn) {
    try {
        const response = await fetch('/api/favorites/toggle', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({topic})
        });
        const data = await response.json();
        btn.innerHTML = data.is_favorite ? '⭐' : '☆';
        btn.style.color = data.is_favorite ? '#fbbf24' : '#6b7280';
    } catch (error) {
        console.error('Favorite error:', error);
    }
}

async function reviewTopic(topic) {
    try {
        await fetch(`/api/review/${encodeURIComponent(topic)}`, {method: 'POST'});
        addMessage(`📚 Topic "${topic}" marked for review!`, false);
    } catch (error) {
        console.error('Review error:', error);
    }
}

// Load history on page load
loadChatHistory();

// Focus input on load
messageInput.focus();

// Dark mode toggle
document.getElementById('darkModeToggle').addEventListener('click', async () => {
    try {
        const response = await fetch('/api/theme/toggle', {method: 'POST'});
        const data = await response.json();
        document.body.classList.toggle('dark-mode', data.dark_mode);
        document.getElementById('themeIcon').textContent = data.dark_mode ? '☀️' : '🌙';
    } catch (error) {
        console.error('Theme toggle error:', error);
    }
});

// Load theme on page load
fetch('/api/theme')
    .then(r => r.json())
    .then(data => {
        if (data.dark_mode) {
            document.body.classList.add('dark-mode');
            document.getElementById('themeIcon').textContent = '☀️';
        }
    });

// Load current model
fetch('/api/models/current')
    .then(r => r.json())
    .then(data => {
        if (data.model) {
            document.getElementById('modelSelect').value = data.model;
        }
    });

// Model selection
document.getElementById('modelSelect').addEventListener('change', async (e) => {
    const model = e.target.value;
    try {
        const response = await fetch('/api/models/select', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model})
        });
        const data = await response.json();
        if (data.success) {
            addMessage(`🤖 Switched to ${model}`, false);
        }
    } catch (error) {
        console.error('Model selection error:', error);
    }
});


// Image upload functionality
const imageBtn = document.getElementById('imageBtn');
const imageInput = document.getElementById('imageInput');

imageBtn.addEventListener('click', () => {
    imageInput.click();
});

imageInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = async (event) => {
        const imageData = event.target.result;
        
        const previewDiv = document.createElement('div');
        previewDiv.className = 'message user-message';
        previewDiv.innerHTML = `
            <div class="message-content">
                <p>📷 Uploaded image for analysis</p>
                <img src="${imageData}" style="max-width: 300px; border-radius: 8px; margin-top: 0.5rem;">
            </div>
        `;
        chatMessages.appendChild(previewDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        addMessage('Analyzing image...', false);
        
        try {
            const response = await fetch('/api/image/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: imageData})
            });
            
            const data = await response.json();
            chatMessages.removeChild(chatMessages.lastChild);
            
            if (data.error) {
                addMessage(`Error: ${data.error}`, false);
            } else {
                if (data.extracted_text && data.extracted_text !== '[No text detected in image]') {
                    addMessage(`📝 Extracted Text:\n${data.extracted_text}`, false);
                }
                addMessage(`🤖 Analysis:\n${data.analysis}`, false);
            }
        } catch (error) {
            chatMessages.removeChild(chatMessages.lastChild);
            addMessage('Error analyzing image. Please try again.', false);
            console.error('Image analysis error:', error);
        }
    };
    
    reader.readAsDataURL(file);
    imageInput.value = '';
});


// Voice Input
let recognition = null;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onresult = async (event) => {
        const transcript = event.results[0][0].transcript;
        messageInput.value = transcript;
        document.getElementById('voiceBtn').innerHTML = '🎤';
        
        // Auto-submit to RAG
        if (transcript.trim()) {
            addMessage(transcript, true);
            messageInput.value = '';
            addMessage('Thinking...', false);
            
            try {
                const response = await fetch('/api/chat/message', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: transcript})
                });
                
                const data = await response.json();
                chatMessages.removeChild(chatMessages.lastChild);
                addMessage(data.response, false);
                
                if (data.topic) {
                    addMessageActions(chatMessages.lastChild, data.topic);
                }
                
                if (data.external_resources) {
                    addExternalResources(data.external_resources);
                }
                
                if (data.is_new_topic === true && data.topic) {
                    const quizPrompt = document.createElement('div');
                    quizPrompt.className = 'message bot-message quiz-offer';
                    quizPrompt.innerHTML = `
                        <div class="message-content">
                            <p style="font-weight: bold; color: #6366f1;">🎯 This is a new topic for you!</p>
                            <p>Would you like to take a quick quiz to test your understanding?</p>
                            <div style="margin-top: 1rem; display: flex; gap: 0.5rem;">
                                <button onclick="takeQuickQuiz('${data.topic}')" class="btn btn-primary">✅ Yes, Take Quiz</button>
                                <button onclick="this.closest('.quiz-offer').remove()" class="btn btn-secondary">❌ No, Thanks</button>
                            </div>
                        </div>
                    `;
                    chatMessages.appendChild(quizPrompt);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                // Auto-play response
                const lastMessage = chatMessages.lastChild;
                const playBtn = lastMessage.querySelector('.btn-icon');
                if (playBtn) {
                    setTimeout(() => speakText(data.response, playBtn), 500);
                }
                
            } catch (error) {
                chatMessages.removeChild(chatMessages.lastChild);
                addMessage('Sorry, something went wrong. Please try again.', false);
                console.error('Voice chat error:', error);
            }
        }
    };
    
    recognition.onerror = () => {
        document.getElementById('voiceBtn').innerHTML = '🎤';
    };
    
    recognition.onend = () => {
        document.getElementById('voiceBtn').innerHTML = '🎤';
    };
}

document.getElementById('voiceBtn').addEventListener('click', () => {
    if (!recognition) {
        alert('Voice input not supported in this browser');
        return;
    }
    
    const btn = document.getElementById('voiceBtn');
    if (btn.innerHTML === '🔴') {
        recognition.stop();
        btn.innerHTML = '🎤';
    } else {
        recognition.start();
        btn.innerHTML = '🔴';
    }
});

// Web Search
document.getElementById('webSearchBtn').addEventListener('click', async () => {
    const query = messageInput.value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    addMessage(query, true);
    messageInput.value = '';
    addMessage('🔍 Searching the web...', false);
    
    try {
        const response = await fetch('/api/search/web', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query})
        });
        
        const data = await response.json();
        chatMessages.removeChild(chatMessages.lastChild);
        
        if (data.results && data.results.length > 0) {
            let resultsHtml = `<h3>🔍 Web Search Results</h3>`;
            resultsHtml += `<p><strong>Query:</strong> "${data.original_query}"</p>`;
            resultsHtml += `<p><strong>Keywords:</strong> ${data.query}</p>`;
            resultsHtml += `<p><strong>Found:</strong> ${data.count} results</p>`;
            
            // Add images at the top if available
            if (data.images && data.images.length > 0) {
                addImageGallery(data.images);
                
                // Save images separately
                const imagesJson = JSON.stringify(data.images);
                await fetch('/api/chat/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        query: `[IMAGES]${query}`,
                        response: imagesJson
                    })
                });
            }
            
            resultsHtml += `<hr style="margin: 1rem 0; border: none; border-top: 1px solid #ddd;">`;
            
            data.results.forEach((result, index) => {
                resultsHtml += `
                    <div style="margin: 1rem 0; padding: 1rem; background: #f5f5f5; border-radius: 8px; border-left: 3px solid #6366f1;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">${index + 1}. ${result.title}</h4>
                        <p style="margin: 0.5rem 0; color: #555; font-size: 0.95rem;">${result.snippet}</p>
                        <div style="margin-top: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                            <span style="background: #6366f1; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">${result.source}</span>
                            <a href="${result.url}" target="_blank" style="color: #6366f1; text-decoration: none; font-size: 0.9rem;">
                                🔗 Open Link →
                            </a>
                        </div>
                    </div>
                `;
            });
            
            resultsHtml += `<div style="margin-top: 1rem; padding: 0.75rem; background: #e0f2fe; border-radius: 6px; font-size: 0.9rem;">`;
            resultsHtml += `<strong>💾 Saved to Knowledge Base</strong><br>`;
            resultsHtml += `This search has been saved to your learning history.`;
            resultsHtml += `</div>`;
            
            addMessage(resultsHtml, false);
            
            // Save to chat history
            await fetch('/api/chat/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query, response: resultsHtml})
            });
        } else {
            const noResultsMsg = 'No results found. Try a different search query.';
            addMessage(noResultsMsg, false);
            
            // Save to chat history
            await fetch('/api/chat/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query, response: noResultsMsg})
            });
        }
    } catch (error) {
        chatMessages.removeChild(chatMessages.lastChild);
        addMessage('Error searching the web. Please try again.', false);
        console.error('Web search error:', error);
    }
});

// Code Playground
document.getElementById('codeBtn').addEventListener('click', () => {
    document.getElementById('codeModal').style.display = 'block';
});

document.getElementById('closeModal').addEventListener('click', () => {
    document.getElementById('codeModal').style.display = 'none';
});

document.getElementById('runCodeBtn').addEventListener('click', async () => {
    const code = document.getElementById('codeEditor').value;
    const language = document.getElementById('languageSelect').value;
    const outputDiv = document.getElementById('codeOutput');
    
    if (!code.trim()) {
        outputDiv.style.display = 'block';
        outputDiv.innerHTML = '<span style="color: red;">Error: No code to execute</span>';
        return;
    }
    
    outputDiv.style.display = 'block';
    outputDiv.innerHTML = 'Running...';
    
    try {
        const response = await fetch('/api/code/execute', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({code, language})
        });
        
        const data = await response.json();
        
        if (data.error && data.error.trim()) {
            outputDiv.innerHTML = `<span style="color: red;">Error:\n${data.error}</span>`;
        } else if (data.output) {
            outputDiv.innerHTML = `<span style="color: green;">Output:\n${data.output}</span>`;
        } else {
            outputDiv.innerHTML = '<span style="color: gray;">No output</span>';
        }
    } catch (error) {
        outputDiv.innerHTML = `<span style="color: red;">Error: ${error.message}</span>`;
    }
});
