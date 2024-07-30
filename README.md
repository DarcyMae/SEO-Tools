# Google Review Analyzer and Responder
This project combines a pre-trained LSTM model and Claude AI to analyze Google reviews, determine their sentiment, and generate appropriate responses. It uses a Gradio interface for easy interaction.
## Features
- Sentiment analysis of Google reviews using a pre-trained LSTM model
- AI-powered response generation using Claude 3.5 Sonnet
- User-friendly interface with Gradio
- Comparison of LSTM and Claude sentiment analysis results
## Use Cases
1. Customer Service: Quickly analyze and respond to customer reviews
2. Reputation Management: Monitor and address both positive and negative feedback
3. Business Insights: Gain understanding of customer sentiments and common issues
4. Training Tool: Help customer service representatives craft appropriate responses
## Requirements
- Python 3.10+
- TensorFlow
- Anthropic API
- Gradio
- langchain
- python-dotenv
Install the required packages using:

```pip install tensorflow anthropic gradio langchain python-dotenv```
##  Setup
1. Clone the repository
2. Create a ```.env```file with you Anthropic API key:
```ClaudeAPI_Key=your_api_key_here```
3. Ensure you have the pre-trained model (```SecondRun.keras```) and tokenizer (```tokenizer.pickle```) in the project directory.
## Usage

Run the Jupyter notebook or extract the code into a Python script. The main components are:

1. Loading the pre-trained LSTM model and tokenizer:

```python
model = tf.keras.models.load_model('SecondRun.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
```
2. Analyzing and responding to reviews:

```python
def analyze_and_respond(review):
    # Tokenize and pad the input review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=50)
    
    # Get prediction from your model
    lstm_prediction = model.predict(padded_sequence)[0][0]
    lstm_sentiment = "positive" if lstm_prediction > 0.5 else "negative"
    
    # Use Claude to analyze and respond
    # ... (see full function in the notebook)
```
3. Gradio interface:

```python
with gr.Blocks() as demo:
    gr.Markdown("# Google Review Analyzer and Responder")
    
    review_input = gr.Textbox(label="Google Review")
    analyze_button = gr.Button("Analyze Review")
    
    final_output = gr.Textbox(label="Analysis and Response")
    
    with gr.Group(visible=False) as approval_group:
        approve_button = gr.Button("Approve Response")
    
    analyze_button.click(
        process_review,
        inputs=[review_input],
        outputs=[final_output, approval_group]
    )

demo.launch()
```
## How It Works
1. User inputs a Google review
2. The LSTM model redicts the sentiment
3. Claude AI analyzes the review and generates a response
4. The system compares LSTM and Claude sentiments
5. The final output displays the agreement status and the generated response

## Visualization
While the current code doesn't include visualizations, you could enhance the project by adding:
1. A bar chart comparing LSTM and Claude sentiment confidence scores
2. Word clouds of common terms in positive and negative reviews
3. A pie chart showing the distribution of positive vs. negative reviews
To implement these, consider using libraries like matplotlib or plotly, and integrate them into the Gradio interface.
## Future Improvements
- Fine-tune the LSTM model on more recent data
- Implement multi-language support
-Add user feedback mechanism to improve response quality
- Integrate with actual Google Business API for automated responses
## Contributors
- Darcy DeBord
- Aaron Cranor
- Stephen Martinez
