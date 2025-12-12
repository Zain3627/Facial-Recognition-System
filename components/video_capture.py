"""Custom Streamlit component for continuous video capture."""

import streamlit.components.v1 as components

def video_capture_component(capture_interval_ms: int = 2000, height: int = 500) -> str | None:
    """
    Renders a video element that automatically captures frames and returns them as base64.
    
    Args:
        capture_interval_ms: Milliseconds between frame captures
        height: Height of the component in pixels
        
    Returns:
        Base64 encoded JPEG image data, or None if no frame captured yet
    """
    
    html_code = f"""
    <div id="video-container" style="text-align: center; font-family: sans-serif;">
        <video id="webcam" autoplay playsinline muted
            style="width: 100%; max-width: 640px; border-radius: 12px; border: 3px solid #4CAF50; background: #000;">
        </video>
        <canvas id="canvas" style="display: none;"></canvas>
        <div id="status" style="margin-top: 10px; padding: 8px; border-radius: 6px; background: #e8f5e9; color: #2e7d32; font-weight: 500;">
            🎥 Initializing camera...
        </div>
    </div>
    
    <script>
    (function() {{
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');
        const status = document.getElementById('status');
        const ctx = canvas.getContext('2d');
        let frameCount = 0;
        let lastSentData = null;
        
        // Request camera access
        navigator.mediaDevices.getUserMedia({{ 
            video: {{ 
                width: {{ ideal: 640 }}, 
                height: {{ ideal: 480 }},
                facingMode: 'user'
            }}, 
            audio: false 
        }})
        .then(stream => {{
            video.srcObject = stream;
            video.play();
            status.innerHTML = '🔴 <b>LIVE</b> - Capturing frames every {capture_interval_ms // 1000}.{(capture_interval_ms % 1000) // 100}s';
            status.style.background = '#ffebee';
            status.style.color = '#c62828';
            
            // Start capturing frames at regular intervals
            setInterval(() => {{
                if (video.readyState === video.HAVE_ENOUGH_DATA) {{
                    // Set canvas size to video size
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    
                    // Draw current video frame to canvas
                    ctx.drawImage(video, 0, 0);
                    
                    // Convert to base64 JPEG (0.8 quality for speed)
                    const imageData = canvas.toDataURL('image/jpeg', 0.8);
                    
                    // Only send if different from last frame (avoid duplicates)
                    if (imageData !== lastSentData) {{
                        lastSentData = imageData;
                        frameCount++;
                        
                        // Send to Streamlit
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue',
                            value: imageData
                        }}, '*');
                        
                        // Update status
                        const now = new Date().toLocaleTimeString();
                        status.innerHTML = '🔴 <b>LIVE</b> - Frame #' + frameCount + ' captured at ' + now;
                    }}
                }}
            }}, {capture_interval_ms});
        }})
        .catch(err => {{
            status.innerHTML = '❌ Camera error: ' + err.message;
            status.style.background = '#ffebee';
            status.style.color = '#c62828';
            console.error('Camera error:', err);
        }});
    }})();
    </script>
    """
    
    # Use Streamlit's component mechanism
    result = components.html(html_code, height=height)
    
    return result
