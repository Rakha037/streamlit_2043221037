import streamlit as st
import numpy as np
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import binom
import matplotlib.pyplot as plt
import math

st.set_option('deprecation.showPyplotGlobalUse', False)

def binomial_probability(n, x, p):
    coefficient = math.comb(n, x)
    probability = coefficient * (p ** x) * ((1 - p) ** (n - x))
    return probability

def plot_binomial_distribution(n, p):
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)
    fig, ax = plt.subplots()
    ax.bar(x, pmf)
    ax.set_xlabel('X')
    ax.set_ylabel('Probability')
    ax.set_title(f'Binomial Distribution (n={n}, p={p})')
    st.pyplot(fig)

def bernoulli_probability(p, x):
    return p ** x * (1 - p) ** (1 - x)

def normal_probability(x, mean, std):
    z_score = (x - mean) / std
    probability = (1 + np.math.erf(z_score / np.sqrt(2))) / 2
    return probability

def plot_normal_distribution(mean, std):
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    plt.plot(x, y)
    plt.title("Normal Distribution")
    plt.xlabel("X")
    plt.ylabel("Probability Density")
    st.pyplot()

def plot_poisson_pdf(mu, x):
    x = np.arange(0, x+1)
    pmf = poisson.pmf(x, mu)
    pdfx, ax = plt.subplots()
    ax.bar(x, pmf)
    ax.set_xlabel('X')
    ax.set_ylabel('Probability')
    ax.set_title(f'Poisson Distribution PDF (mu={mu})')
    st.pyplot(pdfx)
    
def plot_poisson_cdf(mu, x):
    x = np.arange(0, x+1)
    cdf = poisson.cdf(x, mu)
    cdfx, ax = plt.subplots()
    ax.plot(x, cdf, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Poisson Distribution CDF (mu={mu})')
    st.pyplot(cdfx)

st.markdown(
    """
    <style>
    .small-font {
        font-size: 14px;
        

    }

    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    tipe = st.radio('pilih tipe', ['Distribusi Bernoulli','Distribusi Binomial' , 'Distribusi Poisson', 'Distribusi Normal','Mean, Median, dan Standart Deviasi'])    

if tipe == 'Distribusi Bernoulli':
    st.title("Distribusi Bernoulli")
    with st.form('Submit Dist'):
        p = st.number_input("Masukkan Nilai P(probabilitas tingkat keberhasilan):", value=0.0, step=0.01, max_value=1.0)
        x = st.number_input("Masukkan Nilai X: ", value=0, step=1, max_value=1)
        if st.form_submit_button('Calculate'):
            prob = bernoulli_probability(p, x)
            st.write("Probabilitas peluang dalam distribusi Bernoulli:", prob)

elif tipe == 'Distribusi Binomial':
    st.title("Distribusi Binomial")
    with st.form('Submit Dist'):
        n = st.number_input("Masukkan Jumlah Percobaan (n): ", value=1, step=1, min_value=0)
        x = st.number_input("Masukkan Jumlah Keberhasilan (x)): ", value=1, step=1, min_value=0)
        p = st.number_input("Masukkan Probabilitas Keberhasilan : ", value=0.01, step=0.01)
        if st.form_submit_button('Calculate'):
            prob_b = binomial_probability(n, x, p)
            st.write("Probabilitas peluang dalam distribusi Binomial:", prob_b)
            plot_binomial_distribution(n, p)
            
elif tipe == 'Distribusi Poisson':
    with st.form('Submit Dist'):
        st.title("Perhitungan Distribusi Poisson")
        mu = st.number_input("Masukkan nilai lambda: ", value=1.0, step=0.1)
        x = st.number_input("Masukkan jumlah percobaan: ", value=1, step=1)
        oprasi = st.radio('Pilih Oprasi',['PMF', 'CDF'])
        pdf = poisson.pmf(x, mu)
        cdf = poisson.cdf(x, mu)
        if st.form_submit_button('Calculate'):
            if oprasi == 'PMF':    
                st.write("Hasil Perhitungan:")
                st.write("Peluang distribusi Poisson:", pdf)
                plot_poisson_pdf(mu, x)

            elif oprasi =='CDF':
                st.write("Hasil Perhitungan:")
                st.write("Peluang distribusi Poisson:", cdf)
                plot_poisson_cdf(mu, x)
                
                
                
elif tipe == 'Distribusi Normal':
    st.title("Perhitungan Distribusi Normal")
    col1, col2 = st.columns([1, 1])
    with col1:
        image_url = "https://caraharian.com/wp-content/uploads/2022/03/tabel-z.jpg"
        st.image(image_url,width=300, caption="Pengertian Distribusi Normal" )
    with col2:
        st.markdown('<p class="small-font">Distribusi normal merupakan fungsi probabilitas yang menunjukkan adanya distribusi atau penyebaran suatu variabel. Fungsi dari distribusi normal tersebut biasanya dibuktikan oleh adanya grafik simetris yang disebut kurva lonceng atau bell curve yang menandakan adanya distribusi yang merata, sehingga kurva akan memuncak di bagian tengah dan melandai di kedua sisi dengan nilai yang setara.</p>', unsafe_allow_html=True) 
        st.markdown('<p class="small-font"></p>', unsafe_allow_html=True)
    with st.form('Submit Dist'):
        x = st.number_input("Masukkan nilai x: ", value=0.0, step=0.1)
        mean = st.number_input("Masukkan nilai mean: ", value=0.0, step=0.1)
        std = st.number_input("Masukkan nilai standar deviasi: ", value=0.0, step=0.1)
        if st.form_submit_button('Calculate'):
            dist_norm = normal_probability(x, mean, std)
            st.write("Probabilitas dalam distribusi normal:", dist_norm)
            plot_normal_distribution(mean, std)
elif tipe == 'Mean, Median, dan Standart Deviasi':
    st.title("Perhitungan Mean, Median, dan Standart Deviasi")
    df1, df2 = st.columns(2)
    with df1:
        st.write("Masukkan data anda")
        input_data = st.text_area("Data", height=200)
        
    with df2:
        st.write("Berikut hasilnya")
        if st.button("Hitung"):
            data_list = input_data.split()
            processed_data = [int(val) for val in data_list]
            st.write("Descriptive Statistics:")
            st.write("Mean:", np.mean(processed_data))
            st.write("Standard Deviation:", np.std(processed_data))
            st.write("Minimum:", np.min(processed_data))
            st.write("Maximum:", np.max(processed_data))
        