import pandas as pd
from decision_tree_model import load_and_prepare_data, train_decision_tree

# Load & latih model
X_train, X_test, y_train, y_test = load_and_prepare_data("data/StudentsPerformance.csv")
model = train_decision_tree(X_train, y_train)

# Evaluasi akurasi model
accuracy = model.score(X_test, y_test)
print(f"Akurasi model pada data uji: {accuracy:.2%}")

# Simpan kolom pelatihan
train_columns = X_train.columns

print("\n=== Prediksi Kelulusan Mahasiswa ===")
# Input user
math = int(input("Masukkan nilai Matematika: "))
reading = int(input("Masukkan nilai Membaca: "))
writing = int(input("Masukkan nilai Menulis: "))

gender = input("Jenis Kelamin (male/female): ")
race = input("Kelompok Etnis (group A/B/C/D/E): ")
education = input("Pendidikan Orang Tua (high school, bachelor's degree, etc): ")
lunch = input("Jenis Makan Siang (standard/free-reduced): ")
prep = input("Persiapan Tes (completed/none): ")

# Buat input data frame
user_data = {
    'gender_female': 1 if gender.lower() == 'female' else 0,
    'race/ethnicity_group B': 1 if race == 'group B' else 0,
    'race/ethnicity_group C': 1 if race == 'group C' else 0,
    'race/ethnicity_group D': 1 if race == 'group D' else 0,
    'race/ethnicity_group E': 1 if race == 'group E' else 0,
    'parental level of education_high school': 1 if education == 'high school' else 0,
    'parental level of education_some college': 1 if education == 'some college' else 0,
    'parental level of education_associate\'s degree': 1 if education == "associate's degree" else 0,
    'parental level of education_bachelor\'s degree': 1 if education == "bachelor's degree" else 0,
    'parental level of education_master\'s degree': 1 if education == "master's degree" else 0,
    'lunch_standard': 1 if lunch == 'standard' else 0,
    'test preparation course_completed': 1 if prep == 'completed' else 0,
    'math score': math,
    'reading score': reading,
    'writing score': writing
}

# Pastikan semua kolom ada dan urutannya sesuai dengan kolom pelatihan
user_df = pd.DataFrame([user_data])
user_df = user_df.reindex(columns=train_columns, fill_value=0)

# Prediksi
prediction = model.predict(user_df)[0]

print("\n📊 Hasil Prediksi:")
print("Status: LULUS ✅" if prediction == 1 else "Status: TIDAK LULUS ❌")
print("\nTerima kasih telah menggunakan sistem prediksi kelulusan mahasiswa!")