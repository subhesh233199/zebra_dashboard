function processMarkdownContent(content) {
    if (typeof content !== 'string') {
        console.error('processMarkdownContent: Input is not a string:', content);
        return '<p>Error: Invalid report content</p>';
    }
    
    // First parse the markdown to HTML
    let html = marked.parse(content);
    
    // Enhance risk indicators and trends with proper styling
    html = html.replace(/<strong>HIGH RISK<\/strong>/gi, '<span class="status-high">HIGH RISK</span>');
    html = html.replace(/<strong>MEDIUM RISK<\/strong>/gi, '<span class="status-medium">MEDIUM RISK</span>');
    html = html.replace(/<strong>LOW RISK<\/strong>/gi, '<span class="status-low">LOW RISK</span>');
    html = html.replace(/<strong>ON TRACK<\/strong>/gi, '<span class="status-low">ON TRACK</span>');
    html = html.replace(/<strong>(↑ \([^<]+\))<\/strong>/gi, '<span class="trend-up">$1</span>');
    html = html.replace(/<strong>(↓ \([^<]+\))<\/strong>/gi, '<span class="trend-down">$1</span>');
    html = html.replace(/<strong>→<\/strong>/gi, '<span class="trend-neutral">→</span>');
    
    // Ensure tables are responsive
    html = html.replace(/<table>/g, '<div class="table-responsive"><table class="table table-bordered">');
    html = html.replace(/<\/table>/g, '</table></div>');
    
    return html;
}
