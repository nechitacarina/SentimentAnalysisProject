import tkinter as tk
from tkinter import Label, Entry, Button, Text, Scrollbar

from SentimentAnalysis import clean_and_tokenize_data, vectorizer, logistic_regression_model, svm_model


class SentimentAnalysisInterface(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sentiment Analysis")
        self.geometry("800x600")

        self.label_input_text = Label(self, text="Enter text for sentiment analysis:")
        self.label_input_text.pack(pady=10)

        self.entry_input_text = Entry(self, width=50)
        self.entry_input_text.pack(pady=10)

        self.button_analyze_sentiment = Button(self, text="Analyze Sentiment", command=self.analyze_sentiment)
        self.button_analyze_sentiment.pack(pady=10)

        self.text_output = Text(self, wrap=tk.WORD, width=60, height=10)
        self.text_output.pack(pady=10)

        self.scrollbar_output = Scrollbar(self, command=self.text_output.yview)
        self.scrollbar_output.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output.config(yscrollcommand=self.scrollbar_output.set)

    def analyze_sentiment(self):
        input_text = self.entry_input_text.get()
        cleaned_input = clean_and_tokenize_data(input_text)
        input_vector = vectorizer.transform([cleaned_input])
        lr_prediction = logistic_regression_model.predict(input_vector)
        svm_prediction = svm_model.predict(input_vector)

        result = "Result with logistic regression:\n"
        if lr_prediction == -1:
            result += "Sentiment: Negative\n"
        elif lr_prediction == 1:
            result += "Sentiment: Positive\n"
        elif lr_prediction == 0:
            result += "Sentiment: Neutral\n"

        result += "\nResult with SVM:\n"
        if svm_prediction == -1:
            result += "Sentiment: Negative\n"
        elif svm_prediction == 1:
            result += "Sentiment: Positive\n"
        elif svm_prediction == 0:
            result += "Sentiment: Neutral\n"

        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, result)


if __name__ == "__main__":
    app = SentimentAnalysisInterface()
    app.mainloop()
