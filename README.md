# Çok Amaçlı QoS Rotalama

Bir ağ üzerinde kaynak ve hedef düğüm arasında **gecikme, güvenilirlik ve kaynak kullanımını birlikte optimize eden** en uygun yolu bulan bir rotalama sistemi. Problem iki farklı yapay zekâ yaklaşımıyla çözülür ve sonuçları karşılaştırılır:

- **Genetik Algoritma (GA):** Turnuva seçimi, tek noktalı çaprazlama ve mutasyon ile yol popülasyonunu nesiller boyunca iyileştirir.
- **Q-Learning:** Ajan, ağ üzerinde dolaşarak her düğümde hangi komşuya gitmenin daha iyi olduğunu ödül/ceza ile öğrenir (ε-greedy keşif).

## Özellikler
- 250 düğüm ve 12.000'den fazla bağlantıdan oluşan ağ (`node.csv`, `link.csv`, `demand.csv`)
- Gecikme, güvenilirlik ve kaynak maliyeti için ayarlanabilir ağırlıklar
- GA (popülasyon, nesil sayısı) ve Q-Learning (episode, α, γ, ε) parametrelerinin arayüzden ayarlanması
- Bulunan yolun etkileşimli ağ görselleştirmesi ve metrik karşılaştırması
- Streamlit tabanlı web arayüzü ve Tkinter tabanlı masaüstü arayüzü

## Çok amaçlı maliyet fonksiyonu
```
Maliyet = w_gecikme · toplam_gecikme + w_güvenilirlik · güvenilirlik_maliyeti + w_kaynak · kaynak_maliyeti
```

## Proje yapısı
| Dosya | Görevi |
|---|---|
| `app2.py` | Streamlit web arayüzü |
| `NetworkUI.py` | Tkinter masaüstü arayüzü |
| `Graph.py`, `Node.py`, `Link.py`, `Demand.py` | Ağ modelinin CSV'lerden yüklenmesi |
| `GeneticAlgorithm.py`, `Chromosome.py` | Genetik Algoritma |
| `QLearning.py` | Q-Learning ajanı |
| `MetricsCalculator.py` | Yol geçerliliği ve çok amaçlı maliyet hesabı |

## Kurulum ve çalıştırma
```bash
pip install -r requirements.txt
streamlit run app2.py        # web arayüzü
python NetworkUI.py          # masaüstü arayüzü
```

## Kullanılan teknolojiler
Python, Genetik Algoritma, Q-Learning, NetworkX, Streamlit, Plotly, Tkinter
