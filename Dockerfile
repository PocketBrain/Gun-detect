FROM mcr.microsoft.com/windows/servercore:ltsc2019

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем модель в контейнер
RUN mkdir weights
COPY weights/model.pt weights/

# Создаем папки для входных и выходных данных
RUN mkdir private/images
RUN mkdir private/labels
RUN mkdir output

# Копируем входные данные в контейнер
COPY images private/images
COPY labels private/labels

# Запускаем приложение
CMD ["python", "solution.py"]

