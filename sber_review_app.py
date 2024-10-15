import kivy
kivy.config.Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
import re
from nltk.corpus import stopwords
import pandas as pd

# Устанавливаем размер окна
Window.size = (360, 640)

# Цветовая схема
PRIMARY_COLOR = [0.0, 0.5, 0.0, 1]  # Зеленый цвет Сбербанка
BACKGROUND_COLOR = [1, 1, 1, 1]  # Белый цвет фона

# Загружаем размеченные данные
df = pd.read_csv('labeled_reviews.csv')
# Определяем категории
categories = df['category'].unique().tolist()

# Словарь ключевых слов для каждой категории с весами
keywords = {
    "Ипотека": (["ипотека", "домклик", "транш", "недвижимость", "залог", "оценка", "первоначальный взнос", "процентная ставка", "страхование ипотеки"], 3),
    "Кредитные карты": (["кредитная карта", "кредитка", "лимит", "проценты", "грейс", "беспроцентный период", "обслуживание карты", "кэшбэк"], 2),
    "Сбербанк Онлайн": (["приложение", "Сбербанк Онлайн", "СБОЛ", "интернет-банк", "мобильный банк", "онлайн перевод", "ошибка приложения", "блокировка онлайн"], 2),
    "Инвестиции": (["инвестиции", "брокер", "ПИФ", "акции", "облигации", "инвестиционный счет", "доходность"], 2),
    "Обслуживание в отделении": (["отделение", "сотрудник", "очередь", "касса", "консультант", "хамство", "некомпетентность", "навязывание услуг"], 1),
    "Банкоматы": (["банкомат", "деньги", "снял", "внес", "карта", "чек", "зажевал", "не выдал", "неисправен", "ошибка банкомата"], 1),
    "Страхование": (["страховка", "страхование", "полис", "выплата", "страховой случай", "страховая премия"], 2),
    "СберСпасибо": (["спасибо", "бонусы", "сберспасибо", "программа лояльности", "начисление бонусов", "списание бонусов"], 2),
    "Мобильная связь": (["сбермобайл", "связь", "тариф", "мобильный оператор", "мобильная связь", "сим-карта"], 2),
    "Вклады": (["вклад", "проценты по вкладу", "открыть вклад", "закрыть вклад", "срок вклада"], 2),
    "Счета": (["счет", "расчетный счет", "текущий счет", "открыть счет", "закрыть счет", "комиссия за обслуживание"], 2),
    "Платежи и переводы": (["перевод", "платеж", "сбп", "комиссия за перевод", "ошибка перевода", "задержка перевода"], 2),
}

# Загружаем предобученную модель для анализа тональности
sentiment_model = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

# Загружаем модель и токенизатор для Zero-Shot Classification
model_name = "facebook/bart-large-mnli"
classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Инициализируем Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Создаем список стоп-слов для русского языка
russian_stopwords = stopwords.words('russian')

# Функция для предобработки текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmas = [token.lemma for token in doc.tokens if token.pos != 'PUNCT' and token.pos != 'NUM' and token.lemma is not None]
    return ' '.join(lemmas)

# Функция для классификации отзыва на основе правил с весами
def classify_by_rules(review, keywords):
    category_scores = {}
    for category, (words, weight) in keywords.items():
        score = sum([weight for word in words if word in review.lower()])
        if score > 0:
            category_scores[category] = score
    if not category_scores:
        return "Другие"
    return max(category_scores, key=category_scores.get)

class ReviewAnalyzerApp(App):
    def build(self):
        # Создаем главный layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Фон
        with layout.canvas.before:
            Color(*BACKGROUND_COLOR)
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
            layout.bind(size=self._update_rect, pos=self._update_rect)

        # Добавляем заголовок
        layout.add_widget(Label(text="Анализатор отзывов о Сбербанке", font_size='24sp', color=PRIMARY_COLOR, size_hint_y=None, height=50))

        # Добавляем поле для ввода отзыва
        self.review_input = TextInput(multiline=True, size_hint_y=None, height=300, hint_text="Введите ваш отзыв", background_color=BACKGROUND_COLOR, foreground_color=[0, 0, 0, 1])
        layout.add_widget(self.review_input)

        # Добавляем кнопку для анализа
        analyze_button = Button(text="Анализировать", size_hint_y=None, height=50, background_color=PRIMARY_COLOR, color=[1, 1, 1, 1])
        analyze_button.bind(on_press=self.analyze_review)
        layout.add_widget(analyze_button)

        # Добавляем метку для вывода результата
        self.result_label = Label(text="", font_size='18sp', color=[0, 0, 0, 1], size_hint_y=None, height=150)
        layout.add_widget(self.result_label)

        return layout

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def analyze_review(self, instance):
        # Получаем текст отзыва
        review_text = self.review_input.text

        # Проводим анализ, если текст не пустой
        if review_text:
            # Определяем сентимент
            sentiment = sentiment_model(review_text)[0]['label']

            # Классифицируем по категориям
            predicted_category = classify_by_rules(review_text, keywords)

            # Обновляем метку результата
            self.result_label.text = f"Сентимент: {sentiment}\nКатегория: {predicted_category}"
        else:
            self.result_label.text = "Пожалуйста, введите отзыв."

# Запускаем приложение, если скрипт выполняется напрямую
if __name__ == '__main__':
    ReviewAnalyzerApp().run()
