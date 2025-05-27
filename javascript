function processMarkdownContent(content) {
    if (typeof content !== 'string') {
        console.error('processMarkdownContent: Input is not a string:', content);
        return '<p>Error: Invalid report content</p>';
    }
    
    // First parse the markdown to HTML
    let html = marked.parse(content);
    
    // Enhance status indicators with proper styling
    html = html.replace(/ON TRACK/gi, '<span class="status-on-track">ON TRACK</span>');
    html = html.replace(/MEDIUM RISK/gi, '<span class="status-medium-risk">MEDIUM RISK</span>');
    html = html.replace(/RISK/gi, '<span class="status-risk">RISK</span>');
    
    // Enhance trend indicators
    html = html.replace(/(↑ \([^<]+\))/gi, '<span class="trend-up">$1</span>');
    html = html.replace(/(↓ \([^<]+\))/gi, '<span class="trend-down">$1</span>');
    html = html.replace(/→/gi, '<span class="trend-stable">→</span>');
    
    // Make tables responsive
    html = html.replace(/<table>/g, '<div class="table-responsive"><table class="table table-bordered">');
    html = html.replace(/<\/table>/g, '</table></div>');
    
    // Add section dividers
    html = html.replace(/<h2>/g, '<div class="section-divider"></div><h2>');
    
    return html;
}
