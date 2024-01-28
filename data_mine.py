# Gerekli kütüphaneleri içe aktarın
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veri setini eğitim ve test veri setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ÖVR Modeli oluşturun (Logistic Regression kullanarak)
ovr_model = LogisticRegression(multi_class='ovr')

# Modeli eğitin
ovr_model.fit(X_train, y_train)

# Test veri setinde tahminler yapın
y_pred = ovr_model.predict(X_test)

# Sınıf olasılıklarını tahmin edin
y_proba = ovr_model.predict_proba(X_test)

# ROC eğrisini ve AUC değerini hesaplayın
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(iris.target_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_proba[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC eğrilerini çizin
plt.figure()
for i in range(len(iris.target_names)):
    plt.plot(fpr[i], tpr[i], label='ROC curve (class %s) (AUC = %0.2f)' % (iris.target_names[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
