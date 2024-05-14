# Image to Story

In this app you can upload an image via the `streamlit` UX and it automatically turn it into an audio story. It covers three main steps to perform the task.

First it uses the image-to-text model `Salesforce/blip-image-captioning-base` to let the machine understand what is the scenario based on the photo. Next we use a LLM by using `lanchain` which will be used to generate a short story and finally we use a text-to-speech model to generate an audio story.

The app uses the UI streamlit for a better user interface.

To run the app:

```python
streamlit run app.py```
