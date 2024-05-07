from django.views.generic import View
from django.shortcuts import render


# class IndexView(View):
#     def get(self, request, *args, **kwargs):
#         return render(request, "app/index.html")

from .forms import ImageUploadForm

from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import os


def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():

            # Renderのフリープランのメモリサイズでも動くように、28*28の軽い画像の処理にする

            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(28, 28))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 28, 28))
            img_array = img_array/255

            # MINISTの判別モデル（学習済み）を動かす

            model_path = os.path.join(settings.BASE_DIR, 'app', 'models', 'model.h5')
            model = load_model(model_path)
            result = model.predict(img_array)

            # 返ってきた結果から、一番確率が高いものをpredictionに入力
            # MNISTだと数値なので、argmax()の値をそのまま入れている
            prediction = result.argmax()

            return render(request, 'home.html', {'form': form, 'prediction': prediction})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})