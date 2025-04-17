// Function to toggle fullscreen for the comparison slider
function toggleCompareFullscreen() {
    console.log('toggleCompareFullscreen called');
    
    // Get the preview column element
    const previewCol = document.getElementById('compare_preview_column');
    if (!previewCol) {
        console.error('Preview column not found');
        return;
    }
    
    // Check if the container has the fullscreen class
    const isFullscreen = previewCol.classList.contains('full_preview');
    
    // Get the buttons
    const fullscreenBtn = document.getElementById('compare_fullscreen_button');
    const downloadBtn = document.getElementById('compare_download_button');
    
    // Get the slider element
    const compareSlider = document.getElementById('compare_slider');
    if (!compareSlider) {
        console.error('Compare slider not found');
        return;
    }
    
    if (isFullscreen) {
        // Exit fullscreen
        previewCol.classList.remove('full_preview');
        if (fullscreenBtn) fullscreenBtn.classList.remove('full');
        if (downloadBtn) downloadBtn.classList.remove('full');
        document.body.style.overflow = 'auto'; // Restore scrolling
        
        // Reset the slider styles
        compareSlider.style.zIndex = "";
        compareSlider.style.height = "500px";
        
        // Reset any image-specific styles
        const images = compareSlider.querySelectorAll('img');
        images.forEach(img => {
            img.style.imageRendering = '';
            img.style.width = '';
            img.style.height = '';
        });
        
        // Apply layout fix after exiting fullscreen
        setTimeout(() => {
            // Force layout recalculation with multiple resize events
            window.dispatchEvent(new Event('resize'));
            
            // Create a temporary element to force layout recalculation
            const tempDiv = document.createElement('div');
            tempDiv.style.position = 'fixed';
            tempDiv.style.top = '0';
            tempDiv.style.right = '0';
            tempDiv.style.width = '1px';
            tempDiv.style.height = '1px';
            tempDiv.style.zIndex = '9999';
            document.body.appendChild(tempDiv);
            
            // Force a layout recalculation
            void tempDiv.offsetHeight;
            
            // Remove after a short delay
            setTimeout(() => {
                document.body.removeChild(tempDiv);
                
                // Additional forceful layout recalculations
                window.dispatchEvent(new Event('resize'));
                
                // Force slider to refresh its own layout
                if (compareSlider) {
                    // Force a reflow by changing slider height temporarily
                    const currentHeight = compareSlider.style.height;
                    compareSlider.style.height = "0";
                    void compareSlider.offsetHeight; // Force reflow
                    compareSlider.style.height = currentHeight;
                    
                    // Adjust the slider container
                    const sliderContainer = compareSlider.querySelector('.slider-wrap');
                    if (sliderContainer) {
                        sliderContainer.style.height = "100%";
                        sliderContainer.style.display = "flex";
                        sliderContainer.style.justifyContent = "center";
                        sliderContainer.style.alignItems = "center";
                    }
                    
                    // Fix slider alignment
                    const sliderHolders = compareSlider.querySelectorAll('.noUi-base, .noUi-connects');
                    sliderHolders.forEach(el => {
                        el.style.position = 'absolute';
                        el.style.top = '50%';
                        el.style.transform = 'translateY(-50%)';
                    });
                    
                    // Fix handle positions
                    const handles = compareSlider.querySelectorAll('.noUi-handle');
                    handles.forEach(handle => {
                        handle.style.position = 'absolute';
                        handle.style.transform = 'translateY(-50%)';
                    });
                }
            }, 100);
        }, 50);
    } else {
        // Enter fullscreen
        previewCol.classList.add('full_preview');
        if (fullscreenBtn) fullscreenBtn.classList.add('full');
        if (downloadBtn) downloadBtn.classList.add('full');
        document.body.style.overflow = 'hidden'; // Prevent scrolling
        
        // Set fullscreen styles for the slider
        compareSlider.style.zIndex = "1000";
        compareSlider.style.height = "90vh";
        
        // Ensure smaller image is upscaled to match larger image using nearest neighbor
        const images = compareSlider.querySelectorAll('img');
        if (images.length === 2) {
            // Find the larger image
            const img1 = images[0];
            const img2 = images[1];
            
            // Determine natural dimensions to see which is bigger
            const img1Area = img1.naturalWidth * img1.naturalHeight;
            const img2Area = img2.naturalWidth * img2.naturalHeight;
            
            if (img1Area > img2Area) {
                // img1 is larger, resize img2 to match
                img2.style.imageRendering = 'pixelated'; // Use nearest-neighbor scaling
                img2.width = img1.naturalWidth;
                img2.height = img1.naturalHeight;
            } else if (img2Area > img1Area) {
                // img2 is larger, resize img1 to match
                img1.style.imageRendering = 'pixelated'; // Use nearest-neighbor scaling
                img1.width = img2.naturalWidth;
                img1.height = img2.naturalHeight;
            }
        }
        
        // Fix for alignment issue using safer layout recalculation methods
        setTimeout(() => {
            // Force layout recalculation with multiple resize events
            window.dispatchEvent(new Event('resize'));
            
            // Create a temporary element to force layout recalculation
            const tempDiv = document.createElement('div');
            tempDiv.style.position = 'fixed';
            tempDiv.style.top = '0';
            tempDiv.style.right = '0';
            tempDiv.style.width = '1px';
            tempDiv.style.height = '1px';
            tempDiv.style.zIndex = '9999';
            document.body.appendChild(tempDiv);
            
            // Force a layout recalculation
            void tempDiv.offsetHeight;
            
            // Remove after a short delay
            setTimeout(() => {
                document.body.removeChild(tempDiv);
                
                // Multiple forceful layout recalculations
                window.dispatchEvent(new Event('resize'));
                
                // Force slider to refresh its own layout
                if (compareSlider) {
                    const currentHeight = compareSlider.style.height;
                    compareSlider.style.height = "0";
                    void compareSlider.offsetHeight; // Force reflow
                    compareSlider.style.height = currentHeight;
                    
                    // Final layout adjustment
                    const sliderContainer = compareSlider.querySelector('.slider-wrap');
                    if (sliderContainer) {
                        sliderContainer.style.height = "100%";
                        sliderContainer.style.display = "flex";
                        sliderContainer.style.justifyContent = "center";
                        sliderContainer.style.alignItems = "center";
                    }
                    
                    // Apply additional styles to fix slider alignment
                    const sliderHolders = compareSlider.querySelectorAll('.noUi-base, .noUi-connects');
                    sliderHolders.forEach(el => {
                        el.style.position = 'absolute';
                        el.style.top = '50%';
                        el.style.transform = 'translateY(-50%)';
                    });
                    
                    // Fix the slider handle positions
                    const handles = compareSlider.querySelectorAll('.noUi-handle');
                    handles.forEach(handle => {
                        handle.style.position = 'absolute';
                        handle.style.transform = 'translateY(-50%)';
                    });
                }
            }, 100);
        }, 50);
    }
}

// Add keyboard event listener for ESC key
document.addEventListener('keydown', function(event) {
    // Check if ESC key was pressed (key code 27)
    if (event.key === 'Escape' || event.keyCode === 27) {
        const previewCol = document.getElementById('compare_preview_column');
        if (previewCol && previewCol.classList.contains('full_preview')) {
            // If the preview column is in fullscreen mode, exit fullscreen
            previewCol.classList.remove('full_preview');
            
            // Get buttons and update their styles
            const fullscreenBtn = document.getElementById('compare_fullscreen_button');
            const downloadBtn = document.getElementById('compare_download_button');
            if (fullscreenBtn) fullscreenBtn.classList.remove('full');
            if (downloadBtn) downloadBtn.classList.remove('full');
            
            // Restore scrolling
            document.body.style.overflow = 'auto';
            
            // Reset slider styles
            const compareSlider = document.getElementById('compare_slider');
            if (compareSlider) {
                compareSlider.style.zIndex = "";
                compareSlider.style.height = "500px";
                
                // Reset any image-specific styles
                const images = compareSlider.querySelectorAll('img');
                images.forEach(img => {
                    img.style.imageRendering = '';
                    img.style.width = '';
                    img.style.height = '';
                });
                
                // Apply layout fix after exiting fullscreen via ESC key
                setTimeout(() => {
                    // Force multiple layout recalculations
                    window.dispatchEvent(new Event('resize'));
                    
                    // Create a temporary element for additional reflow
                    const tempDiv = document.createElement('div');
                    tempDiv.style.position = 'fixed';
                    tempDiv.style.top = '0';
                    tempDiv.style.right = '0';
                    tempDiv.style.width = '1px';
                    tempDiv.style.height = '1px';
                    tempDiv.style.zIndex = '9999';
                    document.body.appendChild(tempDiv);
                    
                    void tempDiv.offsetHeight; // Force reflow
                    
                    setTimeout(() => {
                        document.body.removeChild(tempDiv);
                        window.dispatchEvent(new Event('resize'));
                        
                        // Force slider refresh
                        if (compareSlider) {
                            const currentHeight = compareSlider.style.height;
                            compareSlider.style.height = "0";
                            void compareSlider.offsetHeight;
                            compareSlider.style.height = currentHeight;
                            
                            // Adjust container and elements
                            const sliderContainer = compareSlider.querySelector('.slider-wrap');
                            if (sliderContainer) {
                                sliderContainer.style.height = "100%";
                                sliderContainer.style.display = "flex";
                                sliderContainer.style.justifyContent = "center";
                                sliderContainer.style.alignItems = "center";
                            }
                            
                            // Fix slider components alignment
                            const sliderHolders = compareSlider.querySelectorAll('.noUi-base, .noUi-connects');
                            sliderHolders.forEach(el => {
                                el.style.position = 'absolute';
                                el.style.top = '50%';
                                el.style.transform = 'translateY(-50%)';
                            });
                            
                            const handles = compareSlider.querySelectorAll('.noUi-handle');
                            handles.forEach(handle => {
                                handle.style.position = 'absolute';
                                handle.style.transform = 'translateY(-50%)';
                            });
                        }
                    }, 100);
                }, 50);
            }
        }
    }
}); 