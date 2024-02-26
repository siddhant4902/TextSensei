# TextSensei: A Transformer-Based Market News Summarizer

## Introduction

In today's digital age, the abundance of news and information can be overwhelming. TextSensei, our project, aims to simplify the consumption of news articles by automatically generating concise and coherent abstractive summaries. Leveraging the Pegasus transformer model, fine-tuned for our use case, we provide readers with human-like summaries, enabling them to grasp core insights without delving into lengthy narratives.

## Objective

The project's primary objective is to fine-tune a Pegasus transformer model for abstractive summarization of news articles. By creating a tool that generates concise summaries, we streamline news consumption, enhancing comprehension and enabling well-informed decisions in a data-driven world.

## Input Data

To facilitate fine-tuning, we used news data from the Inshorts web app, retrieved through web scraping. The dataset includes news headlines and articles, totaling 55,000 rows.

## Model Building

We fine-tuned the Pegasus transformer model on a pre-trained checkpoint using a sequence-to-sequence architecture. Pegasus employs Masked Language Modeling (MLM) and Gap Sentence Generation (GSG) as pre-training objectives, achieving state-of-the-art performance on various summarization tasks.

## AutoTokenizer

The AutoTokenizer from Hugging Face Hub was employed for tokenization, selecting the appropriate tokenizer based on the input text type.

## Training Arguments

Hyperparameters were fine-tuned using the TrainingArguments class from the transformers library, allowing customization of training parameters.

## Trainer

The Trainer class streamlined the training process for the transformer model. After fine-tuning, the model was uploaded to the Hugging Face Hub for convenient access.

## Testing

ROUGE score, measuring n-gram overlap between machine-generated and reference summaries, was used for evaluation. Pegasus achieved the highest ROUGE score, making it the model of choice for deployment.

## Deployment

The model is deployed on Streamlit, offering users three input options: entering text, uploading an MP3 file, or providing a YouTube link.

## Conclusion

Fine-tuning the Pegasus model on Inshorts data resulted in a highly effective summarization tool, capturing essential details while maintaining conciseness and coherence.

## Future Scope

1. **Segmentation into Different Categories**: Implement a categorization system for news segmentation.
2. **Personalization and User Preferences**: Incorporate personalized recommendation features based on user preferences.
3. **Multi-Modal Summary Display**: Support multi-modal summary display with text and voice notes.
4. **Real-Time News Fetching and Summarization**: Integrate API-based functionalities for real-time news fetching and summarization.
5. **Collaboration with News Aggregation Platforms**: Establish partnerships with news aggregation platforms for a broader reach.
