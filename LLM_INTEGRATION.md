# LLM-Based Keyword Extraction Integration

## Overview

The keyword extractor has been enhanced with LLM (Large Language Model) integration to provide more intelligent and context-aware keyword extraction from issue descriptions. The system supports both local LLM servers and fallback mechanisms.

## Features

### 🤖 **LLM Integration**
- **Local LLM Support**: Connects to local LLM servers (Ollama, etc.)
- **Intelligent Extraction**: Uses AI to understand context and extract relevant keywords
- **Structured Output**: Returns keywords with confidence scores and metadata
- **Fallback Mechanism**: Automatically falls back to heuristic keyword extraction if LLM is unavailable

### 🔧 **Flexible Configuration**
- **Multiple Models**: Support for various LLM models (Llama2, CodeLlama, Mistral, etc.)
- **Customizable Settings**: Configurable timeout, max tokens, and model parameters
- **Easy Setup**: Simple configuration for different LLM servers

### 📊 **Enhanced Metadata**
- **Keyword Types**: Categorizes keywords (technical, error, component, action, functional)
- **Confidence Scores**: Provides confidence levels for each extracted keyword
- **Extraction Methods**: Tracks whether keywords came from LLM or heuristic extraction
- **Context Information**: Includes surrounding context for each keyword

## Architecture

### Core Components

#### 1. **LLMInterface (Abstract Base Class)**
```python
class LLMInterface(ABC):
    @abstractmethod
    def extract_keywords(self, issue_description: str) -> List[ExtractedKeyword]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
```

#### 2. **LocalLLMInterface**
- Connects to local LLM servers via HTTP API
- Supports Ollama-compatible servers
- Handles JSON parsing and error recovery
- Provides fallback extraction when parsing fails

#### 3. **MockLLMInterface**
- Testing and development purposes
- Always available for consistent testing
- Uses heuristic extraction as mock implementation

#### 4. **Enhanced KeywordExtractor**
- Integrates LLM extraction with heuristic fallback
- Configurable extraction methods
- Automatic fallback handling
- Rich metadata and reporting

## Usage Examples

### Basic Usage

```python
from modules.keyword_extractor import KeywordExtractor

# Create extractor with LLM enabled (default)
extractor = KeywordExtractor(use_llm=True, fallback_to_rules=True)

# Extract keywords
issue_description = "The application crashes with OutOfMemoryError exceptions"
keywords = extractor.extract_keywords(issue_description)

# Access keyword details
for kw in keywords:
    print(f"{kw.keyword} ({kw.keyword_type.value}) - {kw.confidence:.2f} - {kw.extraction_method}")
```

### LLM Configuration

```python
# Configure custom LLM settings
extractor.configure_llm(
    base_url="http://localhost:11434",
    model="codellama",
    timeout=60,
    max_tokens=2000
)

# Check LLM availability
if extractor.is_llm_available():
    print("LLM is ready for keyword extraction")
else:
    print("LLM unavailable, using heuristic extraction")
```

### Advanced Usage

```python
# Get keywords by type
technical_keywords = extractor.get_keywords_by_type(issue_description, KeywordType.TECHNICAL)
error_keywords = extractor.get_keywords_by_type(issue_description, KeywordType.ERROR)

# Get extraction method statistics
methods_used = extractor.get_extraction_methods_used(issue_description)
print(f"LLM keywords: {methods_used.get('llm', 0)}")
print(f"Heuristic keywords: {methods_used.get('fallback', 0)}")

# Get comprehensive summary
summary = extractor.get_keyword_summary(issue_description)
for kw_type, keywords in summary.items():
    if keywords:
        print(f"{kw_type}: {', '.join(keywords)}")
```

## LLM Server Setup

### Ollama Setup

1. **Install Ollama**:
   ```bash
   # Download from https://ollama.ai
   # Or use package manager
   ```

2. **Pull a Model**:
   ```bash
   ollama pull llama2
   ollama pull codellama
   ollama pull mistral
   ```

3. **Start Ollama Server**:
   ```bash
   ollama serve
   ```

4. **Verify Installation**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Custom LLM Server

The system is designed to work with any Ollama-compatible API. For custom servers, ensure they support:

- `GET /api/tags` - List available models
- `POST /api/generate` - Generate responses

## Configuration Options

### KeywordExtractor Parameters

```python
KeywordExtractor(
    llm_interface=None,        # Custom LLM interface
    use_llm=True,              # Enable LLM extraction
    fallback_to_heuristics=True     # Fallback to heuristic extraction if LLM fails
)
```

### LocalLLMInterface Parameters

```python
LocalLLMInterface(
    base_url="http://localhost:11434",  # LLM server URL
    model="llama2",                     # Model name
    timeout=30,                         # Request timeout (seconds)
    max_tokens=1000                     # Max response tokens
)
```

## LLM Prompt Engineering

The system uses carefully crafted prompts to ensure high-quality keyword extraction:

```
You are an expert software engineer analyzing issue descriptions to extract relevant keywords for log analysis.

Issue Description:
{issue_description}

Please extract keywords that would be useful for searching through log files to diagnose this issue. Focus on:
1. Technical terms (APIs, databases, services, etc.)
2. Error types and exception names
3. System components and modules
4. Actions and operations
5. Performance-related terms

Return your response as a JSON object with this structure:
{
    "keywords": [
        {
            "keyword": "keyword_text",
            "type": "technical|error|component|action|functional",
            "confidence": 0.0-1.0,
            "reasoning": "why this keyword is relevant"
        }
    ]
}

Extract 10-15 most relevant keywords. Be specific and technical.
```

## Error Handling

### Connection Errors
- Automatic fallback to heuristic extraction
- Logging of connection issues
- Graceful degradation

### Parsing Errors
- Fallback keyword extraction from response text
- Error logging for debugging
- Continued operation with reduced functionality

### Timeout Handling
- Configurable timeout settings
- Automatic retry with fallback
- User notification of issues

## Performance Considerations

### LLM vs Rule-Based Performance

| Method | Speed | Accuracy | Context Understanding |
|--------|-------|----------|---------------------|
| LLM | Slower | Higher | Excellent |
| Rule-Based | Faster | Good | Limited |

### Optimization Tips

1. **Use Appropriate Models**: Smaller models for faster responses
2. **Configure Timeouts**: Balance between speed and reliability
3. **Cache Results**: Consider caching for repeated extractions
4. **Batch Processing**: Process multiple descriptions together

## Testing

### Test Suite

Run the comprehensive test suite:

```bash
python test_llm_keywords.py
```

### Test Coverage

- ✅ LLM availability testing
- ✅ Keyword extraction with different methods
- ✅ Fallback mechanism validation
- ✅ Configuration testing
- ✅ Error handling verification

### Mock Testing

The system includes a `MockLLMInterface` for testing without requiring a running LLM server:

```python
from modules.keyword_extractor import MockLLMInterface

mock_llm = MockLLMInterface()
extractor = KeywordExtractor(llm_interface=mock_llm, use_llm=True)
```

## Integration with Log Analyzer

The enhanced keyword extractor is fully integrated with the main log analysis system:

```python
from log_analyzer_system import LogAnalysisOrchestrator, AnalysisRequest

# Create orchestrator with LLM-enabled keyword extraction
orchestrator = LogAnalysisOrchestrator(use_llm=True)

# Perform analysis
request = AnalysisRequest(
    log_file_path="app.log",
    issue_description="Application crashes with memory errors"
)

result = orchestrator.analyze_issue(request)

# Access extraction information
print(f"LLM available: {orchestrator.keyword_extractor.is_llm_available()}")
print(f"Extraction methods: {orchestrator.keyword_extractor.get_extraction_methods_used(request.issue_description)}")
```

## Troubleshooting

### Common Issues

1. **LLM Server Not Running**
   - Check if Ollama is running: `ollama list`
   - Verify port 11434 is accessible
   - Check firewall settings

2. **Model Not Found**
   - Pull the required model: `ollama pull llama2`
   - Verify model name in configuration
   - Check available models: `ollama list`

3. **Connection Timeouts**
   - Increase timeout settings
   - Check network connectivity
   - Verify server performance

4. **Parsing Errors**
   - Check LLM response format
   - Verify JSON structure
   - Review prompt engineering

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your keyword extraction
extractor = KeywordExtractor(use_llm=True)
```

## Future Enhancements

### Planned Features

1. **Multiple LLM Support**: Support for different LLM providers
2. **Caching**: Intelligent caching of LLM responses
3. **Batch Processing**: Process multiple descriptions efficiently
4. **Custom Prompts**: User-defined prompt templates
5. **Performance Metrics**: Detailed performance tracking

### Extensibility

The system is designed for easy extension:

- Add new LLM providers by implementing `LLMInterface`
- Customize keyword types by extending `KeywordType` enum
- Modify extraction logic by overriding methods
- Add new confidence calculation algorithms

## Conclusion

The LLM-based keyword extraction provides significant improvements in accuracy and context understanding while maintaining robust fallback mechanisms. The system is production-ready and provides a solid foundation for intelligent log analysis.
