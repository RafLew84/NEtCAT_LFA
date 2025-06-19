# NEtCAT_LFA
 
## Lattice Fourier Analyzer (LFA)
**Lattice Fourier Analyzer (LFA)** to zaawansowana aplikacja naukowa napisana w języku Python, przeznaczona do analizy obrazów ze Skaningowego Mikroskopu Tunelowego (STM). Program umożliwia wyznaczanie parametrów sieci krystalicznych na podstawie analizy obrazów w przestrzeni Fouriera, oferując jednocześnie narzędzia do korekcji zniekształceń i symulacji teoretycznych.

### Instalacja i Uruchomienie
Aby uruchomić aplikację, postępuj zgodnie z poniższymi krokami:

1. Pobranie Kodu
Sklonuj lub pobierz repozytorium na swój lokalny dysk.

2. Instalacja Zależności
Upewnij się, że masz zainstalowanego **Pythona** (w wersji 3.9 lub nowszej). Następnie, w **głównym katalogu projektu**, zainstaluj wszystkie wymagane biblioteki za pomocą przygotowanego pliku `requirements.txt`:

```Bash
pip install -r requirements.txt
```

3. Uruchomienie Aplikacji
Będąc w **głównym katalogu projektu** (domyślnie `NEtCAT_LFA`), uruchom aplikację za pomocą następującej komendy:


```Bash
python -m lfa.main
```

## Opis Aplikacji i Możliwości Obliczeniowe
LFA to interaktywne narzędzie, które prowadzi użytkownika przez cały proces analizy – od surowego obrazu STM po ilościowe wyniki.

### Główne możliwości obliczeniowe:

* **Parametry Sieci Rzeczywistej**: Obliczanie wektorów sieciowych ($a_1, a_2$), ich długości ($|a_1|, |a_2|$) oraz kąta ($\alpha$) między nimi dla substratu i warstw adsorbatu.
* **Korekcja Dryftu i Zniekształceń**: Wyznaczanie macierzy transformacji afinicznej ($F$) i wektora translacji ($t$), które opisują zniekształcenia obrazu (rotację, rozciąganie/ścisk).
* **Analiza Dopasowania Sieci**: Ilościowe określenie jakości dopasowania zmierzonych pików do idealnej siatki za pomocą błędu średniokwadratowego (RMSE).
* **Analiza Ścian Domenowych**: Obliczanie okresowości w przestrzeni rzeczywistej oraz stosunków intensywności/amplitudy pików satelitarnych, co pozwala na charakteryzację nadstruktur.
* **Autokorelacja (Mapa Pattersona)**: Generowanie mapy autokorelacji z widma mocy FFT w celu wizualizacji wektorów sieciowych w przestrzeni rzeczywistej.

### Dostępne Opcje Preprocessingu
Przed główną analizą, jakość obrazu można poprawić za pomocą następujących narzędzi dostępnych w menu `Preprocessing`:
* **Gaussian Blur**: Rozmycie gaussowskie w celu usunięcia szumu wysokoczęstotliwościowego.
* **Gaussian Sharpening**: Wyostrzanie obrazu metodą maski nieostrości.
* **Plane Leveling**: Korekcja nachylenia tła poprzez dopasowanie i odjęcie płaszczyzny.
* **Median Filter**: Filtracja medianowa, skuteczna w usuwaniu szumu typu "sól i pieprz".
* **NL-Means Denoising**: Zaawansowany algorytm odszumiania zachowujący detale obrazu.
* **BM3D Denoising**: Wysokiej jakości, choć obliczeniowo intensywny, algorytm odszumiania.

## Przewodnik Analityczny Krok po Kroku
Typowa sesja analityczna w LFA przebiega następująco:

### 1. Obliczanie FFT
1. Wczytaj plik z danymi STM (`.stp`, `.s94`) poprzez menu `File > Open...`.
2. (Opcjonalnie) Zastosuj wybrane operacje preprocessingu. Każda operacja tworzy nowy element w panelu `History`, z którego można w każdej chwili skorzystać.
3. Wybierz w panelu `History` obraz, dla którego chcesz policzyć FFT.
4. Wybierz z menu `Analysis > Calculate FFT...`. Otworzy się okno dialogowe, które oferuje podgląd na żywo. Po lewej stronie znajduje się oryginalny obraz, a po prawej - wynik transformaty Fouriera.

W oknie tym masz do dyspozycji następujące opcje:

* **Obliczenia na fragmencie (ROI)**: Możesz obliczyć FFT dla całego obrazu lub tylko dla zaznaczonego fragmentu. Aby aktywować ten tryb, zaznacz opcję Calculate FFT only for ROI area. Na obrazie po lewej stronie pojawi się prostokąt, który możesz przesuwać i skalować, aby wybrać interesujący Cię obszar. Jest to przydatne, gdy chcesz przeanalizować lokalne właściwości sieci krystalicznej.

* **Funkcja Okna (Window Function)**: Przed obliczeniem transformaty możesz zastosować funkcję okna (np. hann, hamming), aby zredukować artefakty spektralne (tzw. wyciek widma), które mogą pojawić się na krawędziach obrazu lub zaznaczonego ROI.

* **Skalowanie Wyświetlania (Display Scaling)**: Możesz wybrać sposób wizualizacji widma mocy FFT. Opcja ta ma wpływ tylko na podgląd i ostateczny zapisany obraz, a nie na dane zespolone używane w tle. Dostępne tryby to:
  * **Log Magnitude**: Skala logarytmiczna, najlepsza do uwidocznienia słabych pików obok bardzo intensywnych.
  * **Power Spectrum**: Skala kwadratowa ($∣F∣^2$), która reprezentuje fizyczną intensywność (moc) sygnału. Jest to wymagane ustawienie do analizy intensywności przy badaniu ścian domenowych.
  * **Linear Magnitude**: Skala liniowa ($∣F∣$), bezpośrednia amplituda.
  * **Sqrt Magnitude**: Skala pierwiastkowa ($∣F∣$), kompromis między skalą liniową a logarytmiczną.

Po zatwierdzeniu ustawień przyciskiem `Apply` FFT, w panelu History pojawi się nowy element FFT, a po prawej stronie ukaże się panel `FFT Analysis Tools` gotowy do dalszej analizy.

W grupie `Ideal Lattice Overlay` na panelu `FFT Analysis Tools` możesz nałożyć teoretyczną, idealną siatkę dyfrakcyjną dla wybranego substratu na eksperymentalny obraz FFT.

### 2. Analiza Substratu i Korekcja Dryftu (Transformacja F, t)
W panelu FFT Analysis Tools wybierz z listy Substrate typ analizowanej sieci (np. Au(111)) lub zdefiniuj własną (<Custom Define...>), podając stałą sieciową a_surf.
Kliknij przycisk Select/Edit Substrate Spots....
W nowym oknie, zaznacz wymaganą liczbę pików Bragga (6 dla sieci heksagonalnej, 4 dla kwadratowej), korzystając z opcji dopasowania (np. 2D Gaussian Fit) dla uzyskania subpikselowej dokładności.
Po zaznaczeniu wszystkich punktów, kliknij przycisk Calculate Transformation.
Program obliczy i wyświetli macierz transformacji F, wektor t oraz wynikające z nich parametry fizyczne: kąt rotacji, współczynniki rozciągnięcia i błąd RMSE dopasowania.
1. Analiza Adsorbatu
Wróć do głównego okna. W panelu FFT Analysis Tools przełącz tryb na Adsorbate i utwórz nowy zestaw (Set 1).
Kliknij Select/Edit Current Set Spots....
W nowym oknie zaznacz piki pochodzące od adsorbatu.
Kliknij przycisk Apply Substrate Correction to Adsorbate Spots. Program użyje zapisanej wcześniej transformacji (F, t) do przekształcenia współrzędnych pików adsorbatu do idealnego, nieskorygowanego układu współrzędnych substratu.
Po powrocie do okna głównego, kliknij Calculate Adsorbate Parameters. Program obliczy rzeczywiste parametry sieci adsorbatu (|a1|, |a2|, α).
1. Wizualizacja w Przestrzeni Rzeczywistej
W menu Analysis wybierz Visualize Real Space....
Otwarte zostanie okno, które po lewej stronie pokaże obraz FFT, a po prawej zwizualizuje obliczone wektory sieciowe w przestrzeni rzeczywistej dla substratu i aktywnych zestawów adsorbatu.
W tym oknie można również obliczyć kąt względny między siecią substratu a siecią adsorbatu.
1. Analiza Ścian Domenowych
Upewnij się, że obraz FFT został obliczony w trybie Power Spectrum, co jest wymagane do analizy intensywności.
Z menu Analysis wybierz Analyze Domain Walls....
W oknie dialogowym zaznacz kolejno główny pik Bragga oraz jeden z jego pików satelitarnych.
Program obliczy:
Odległość między pikami w przestrzeni odwrotnej (Δg*).
Okresowość ściany domenowej w przestrzeni rzeczywistej.
Stosunek intensywności, amplitudy i maksymalnej wartości piku satelitarnego do głównego.
Symulacja Obrazów STM/FFT
Narzędzie STM/FFT Simulation... (menu Analysis) pozwala na weryfikację modelu teoretycznego z danymi eksperymentalnymi.

Co można zasymulować?: Można wygenerować idealny obraz STM i jego FFT dla wybranego substratu i adsorbatu, uwzględniając parametry takie jak:
Jednoosiowa kompresja sieci adsorbatu.
Szerokość pasów i szerokość relaksacji dla nadstruktur prążkowych.
Symetria i typ domen (np. Striped, Hexagonal; Heavy, Super Heavy).
Co można wyznaczyć?: Dialog umożliwia automatyczną analizę wygenerowanego widma FFT. Można z niego odczytać stosunki intensywności i amplitudy symulowanych pików satelitarnych do głównych, a następnie porównać je z wartościami uzyskanymi z analizy danych eksperymentalnych.
Rekonstrukcja z Przestrzeni Odwrotnej
Narzędzie Real Space Reconstruction... (menu Analysis) pozwala na analizę odwrotną – od obrazu FFT do przestrzeni rzeczywistej.

Autokorelacja: Oblicza mapę autokorelacji (funkcję Pattersona) z widma mocy FFT. Wynikowy obraz pokazuje wektory translacji w sieci rzeczywistej, co pomaga w identyfikacji okresowości i wektorów bazowych sieci.
Maskowanie (ROI / Spoty): Pozwala użytkownikowi na zaznaczenie wybranych regionów lub konkretnych pików na obrazie FFT. Następnie program wykonuje odwrotną transformację, pokazując, które cechy w przestrzeni rzeczywistej odpowiadają za wybrane częstotliwości. Jest to przydatne do identyfikacji pochodzenia poszczególnych pików w widmie.