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

* **Obliczenia na fragmencie (ROI)**: Możesz obliczyć FFT dla całego obrazu lub tylko dla zaznaczonego fragmentu. Aby aktywować ten tryb, zaznacz opcję `Calculate FFT only for ROI`. Na obrazie po lewej stronie pojawi się prostokąt, który możesz przesuwać i skalować, aby wybrać interesujący Cię obszar.

* **Funkcja Okna (Window Function)**: Przed obliczeniem transformaty możesz zastosować funkcję okna (np. `hann`, `hamming`), aby zredukować artefakty spektralne (tzw. *wyciek widma*), które mogą pojawić się na krawędziach obrazu lub zaznaczonego ROI.

* **Skalowanie Wyświetlania (Display Scaling)**: Możesz wybrać sposób wizualizacji widma mocy FFT. Dostępne tryby to:
  * **Log Magnitude**: Skala logarytmiczna, najlepsza do uwidocznienia słabych pików obok bardzo intensywnych.
  * **Power Spectrum**: Skala kwadratowa ($∣F∣^2$), która reprezentuje fizyczną intensywność (moc) sygnału. Jest to wymagane ustawienie do analizy intensywności przy badaniu ścian domenowych.
  * **Linear Magnitude**: Skala liniowa ($∣F∣$), bezpośrednia amplituda.
  * **Sqrt Magnitude**: Skala pierwiastkowa ($∣F∣$), kompromis między skalą liniową a logarytmiczną.

Po zatwierdzeniu ustawień przyciskiem `Apply` FFT, w panelu History pojawi się nowy element FFT, a po prawej stronie ukaże się panel `FFT Analysis Tools` gotowy do dalszej analizy.

W grupie `Ideal Lattice Overlay` na panelu `FFT Analysis Tools` możesz nałożyć teoretyczną, idealną siatkę dyfrakcyjną dla wybranego substratu na eksperymentalny obraz FFT. Na liście rozwijanej `Substrate` ostatnią opcją jest `<Custom Define...>`, jest to opcja, która otwiera dodatkowe okno dialogowe (`CustomLatticeDialog`), gdzie użytkownik może zdefiniować własny substrat, podając jego nazwę, typ sieci oraz stałą sieciową `a_surf`.

### 2. Analiza Substratu i Korekcja Dryftu (Transformacja F, t)
1. Zaznacz obraz FFT na liście `History` na oknie głównym i kliknij przycisk `Analysis/Select Substrate Spots...`.
2. W nowym oknie wybierz typ sieci (heksagonalna lub kwadratowa) i stałą sieciową (dostępną na liście rozwijanej). ostatnią opcją jest `<Custom Define...>`, która otwiera dodatkowe okno dialogowe (`CustomLatticeDialog`), gdzie użytkownik może zdefiniować własny substrat, podając jego nazwę, typ sieci oraz stałą sieciową `a_surf`.
3. Zaznacz wymaganą liczbę pików Bragga (6 dla sieci heksagonalnej, 4 dla kwadratowej), korzystając z opcji dopasowania dla uzyskania subpikselowej dokładności. Aby rozpocząć zaznaczanie, kliknij w dowolnym miejscu obrazu - pojawi się ROI, które można przesuwać przez *drag & drop*, oraz zmieniać jego rozmiar poprzez przeciągnięcie niewielkiego znacznika w narożniku ROI. Aby zakończyć zaznaczanie piku, kliknij przycisk `Add/Update Spot from ROI`. Możliwe metody dopasowania piku:
    * **Direct Click**: Najprostsza metoda. Pozycja piku jest zapisywana dokładnie w miejscu kliknięcia myszą na obrazie FFT.
    * **Max Pixel**: Bardziej precyzyjna metoda. Po kliknięciu na obrazie pojawia się obszar zainteresowania (ROI). Program automatycznie znajduje piksel o najwyższej intensywności wewnątrz tego ROI i to jego współrzędne są traktowane jako pozycja piku.
    * **2D Gaussian Fit**: Najdokładniejsza metoda, pozwalająca na uzyskanie subpikselowej precyzji. Podobnie jak wyżej, program analizuje dane wewnątrz ROI, ale tym razem dopasowuje do nich dwuwymiarową funkcję Gaussa. Centrum tej funkcji staje się pozycją piku. Jest to zalecana metoda dla precyzyjnych obliczeń.
  
Podczas korzystania z metod `Max Pixel` lub `2D Gaussian Fit`, w oknie dialogowym aktywne stają się podglądy na żywo (`Live Previews`), które pomagają w ocenie wybranego piku:
  * **Podgląd ROI (2D i 3D)**: Pokazuje surowe dane pikseli z wnętrza obszaru ROI w formie obrazu 2D oraz interaktywnego wykresu powierzchniowego 3D.
  * **Podgląd Dopasowania Gaussa (2D i 3D)**: Dostępny tylko w trybie 2D Gaussian Fit. Wyświetla teoretyczny, idealny kształt piku po dopasowaniu funkcji Gaussa. Porównanie tego podglądu z podglądem surowych danych pozwala ocenić jakość dopasowania. **UWAGA** Jeżeli podgląd dopasowania i podgląd ROI są dokładnie takie same (włącznie z tłem) to dopasowanie gaussa zostało zakończone niepowodzeniem - przesuń lub zmień rozmiar ROI.
  
4. Po zaznaczeniu pik pojawi się na liście `Selected Spot Management` - zaznaczając pik na liście możesz go usunąć poprzez kliknięcie `Remove Selected`. Drugą opcją jest usunięcie wszystkich pików przez kliknięcie `Clear All`.
5. Po zaznaczeniu wszystkich punktów, kliknij przycisk `Calculate Transformation`.
Program obliczy i wyświetli macierz transformacji `F`, wektor `t` oraz wynikające z nich parametry fizyczne: kąt rotacji, współczynniki rozciągnięcia i błąd RMSE dopasowania.
6. Kliknięcie `OK` zamknie okno i powrócisz do okna głównego aplikacji, gdzie zaznaczone piki substratu powinny być widoczne na obrazie.
**UWAGA** wyznaczyć piki substratu (i adsorbatu) możesz raz dla wszystkich obrazów FFT znajdujących się na liście `History` - po zmianie obrazu piki będą automatycznie nałożone.

Po powrocie do okna głównego w grupie `Spot Selection` na panelu `FFT Analysis Tools` pokazane są obliczone parametry transformacji. W grupie `Real Space Lattice Parameters -> Substrate` aktywny jest przycisk `Calculate Substrate Parameters` - naciskając go program obliczy parametry substratu na podstawie wyznaczonych pików (długości wektorów $|a_1|, |a_2|$, oraz kąt $\alpha$ pomiędzy tymi wektorami)

### 3. Analiza Adsorbatu
1. Z poziomu głównego okna. W panelu `FFT Analysis Tools` (grupa `Spot Selection`) przełącz tryb na `Adsorbate` i utwórz nowy zestaw (Set 1 jest automatycznie tworzony) - aby utworzyć nowy zestaw rozwiń menu `Current Set` i wybierz `<Add New Set ...>`
2. Następnie z menu `Expected Adsorbate Type` wybierz typ sieci, którego spodziewasz się dla analizowanej warstwy adsorbatu. Wybór ten można zmienić w dowolnym momencie (po przejściu do okna analizy adsorbatu ten wybór można zmienić), nawet po zaznaczeniu punktów, aby zobaczyć, jak wpływa to na wyniki. Jest to kluczowa opcja informująca aplikację, w jaki sposób ma zinterpretować zaznaczone piki adsorbatu w celu obliczenia jego wektorów bazowych sieci odwrotnej ($g_1^∗, g_2^∗$). Wybór odpowiedniego typu pozwala na zastosowanie bardziej precyzyjnych i odpornych na błędy algorytmów. 

Logika obliczeniowa zależy bezpośrednio od wyboru, oraz od liczby zaznaczonych punktów:
  * #### Opcja: `Unknown`
  **Kiedy używać**: Gdy sieć adsorbatu jest nieznana, ma niską symetrię (np. prostokątną skośną) lub gdy chcesz w pełni manualnie zdefiniować wektory bazowe.

  **Jak działa**: Program sortuje wszystkie zaznaczone przez Ciebie punkty według ich odległości od centrum obrazu FFT. Najkrótszy wektor (najbliższy pik) jest wybierany jako pierwszy wektor bazowy $g_1^∗$. Następnie program przeszukuje pozostałe wektory (w kolejności od najkrótszego) i wybiera pierwszy, który nie jest współliniowy z $g_1^∗$. Ten wektor staje się drugim wektorem bazowym $g_2^∗$.

  *Wskazówka*: Dla tej opcji najprościej jest zaznaczyć tylko dwa punkty, które mają być wektorami bazowymi. Program potraktuje je bezpośrednio jako $g_1^∗$ i $g_2^∗$.

  * #### Opcja: `Hexagonal`
  **Kiedy używać**: Gdy spodziewasz się, że sieć adsorbatu ma symetrię heksagonalną.

  **Jak działa (dla 6 punktów)**: Jest to najbardziej precyzyjna metoda dla sieci heksagonalnych. Jeśli zaznaczysz 6 pików tworzących sześciokąt, program zastosuje zaawansowaną metodę uśredniania w celu minimalizacji błędów i korekcji ewentualnej anizotropii (rozciągnięcia sieci):
  1.  Identyfikuje 6 wektorów o najmniejszej długości.
  2.  Uśrednia przeciwległe pary, aby znaleźć trzy główne osie symetrii sześciokąta.
  3.  Wybiera dwie z tych uśrednionych osi, które tworzą kąt najbliższy 60°, jako ostateczne wektory bazowe $g_1^∗$ i $g_2^*$.

  **Jak działa (dla 2-5 punktów)**: Jeśli zaznaczysz mniej niż 6 punktów, algorytm nie może skorzystać z uśredniania symetrii i w tej sytuacji obliczenia są traktowane tak jak dla opcji `Unknown`.

  * #### Opcja: `Square`
  **Kiedy używać**: Gdy spodziewasz się sieci kwadratowej.

  **Jak działa (dla 4 punktów)**: Podobnie jak w przypadku sieci heksagonalnej, zaznaczenie 4 pików tworzących prostokąt/kwadrat pozwala na uśrednienie przeciwległych par w celu precyzyjnego wyznaczenia dwóch, w przybliżeniu prostopadłych, wektorów bazowych.

  **Jak działa (dla 2-3 punktów)**: Przy mniejszej liczbie punktów, stosowana jest prostsza metoda opisana w opcji `Unknown`.

Po wyznaczeniu wektorów $g_1^∗$ i $g_2^∗$ tymi metodami, są one używane do obliczenia finalnych parametrów sieci adsorbatu w przestrzeni rzeczywistej.

3. Kliknij `Anylysis/Select Adsorbate Spots...`. W nowym oknie zaznacz piki pochodzące od adsorbatu (tak samo jak w Analizie Substratu).
4. Aby mieć referencję możesz dodać do obrazu piki teoretycznej sieci idealnej i/lub dofitowane piki substratu za pomocą kontrolek po prawej stronie okna w grupie `Display Options (REference Spots)`.
5. Kliknij przycisk `Apply Substrate Correction to Adsorbate Spots`(Przycisk aktywuje się po wybraniu minimalnej liczby spotów). Program użyje zapisanej wcześniej transformacji ($F$, $t$) do przekształcenia współrzędnych pików adsorbatu do idealnego, nieskorygowanego układu współrzędnych substratu.
Po powrocie do okna głównego, kliknij `Calculate Adsorbate Parameters`. Program obliczy rzeczywiste parametry sieci adsorbatu ($|a1|, |a2|, \alpha$).

### 4. Wizualizacja w Przestrzeni Rzeczywistej
1. W menu `Analysis` wybierz `Visualize Real Space...`. Otwarte zostanie okno, które po lewej stronie pokaże obraz FFT, a po prawej zwizualizuje obliczone wektory sieciowe w przestrzeni rzeczywistej dla substratu i aktywnych zestawów adsorbatu.
2. Naciśnij przycisk `Calculate Sub-Ads Angle` aby obliczyć kąt względny między siecią substratu a siecią adsorbatu.

### 5. Analiza Ścian Domenowych
**UWAGA** Upewnij się, że obraz FFT został obliczony w trybie `Power Spectrum`, co jest wymagane do analizy intensywności.

1. Z menu `Analysis` wybierz `Analyze Domain Walls...`.
2. W oknie dialogowym zaznacz kolejno główny pik Bragga oraz jeden z jego pików satelitarnych. Do wyznaczenia pozycji piku wykorzystywana jest metoda `2D Gaussian Fit`. **UWAGA** Jeżeli jest zaznaczona opcja `Apply Substrate Tranformation`, pozycja każdego piku będzide wyznaczana po obliczeniu transformacji. Jest to odwzorowane przez `x` na obrazie FFT.
  * Aby zaznaczyć głowny pik Bragga kliknij na obraz, co spowoduje pojawienie się ROI - kliknięcie przycisku `Add/Update Main Spot From ROI` zapisze pik główny, kliknięcie `Add Satelite Spot from ROI` zapisze pik satelicki.
  * Po dodaniu piku program obliczy (Wyniki w grupie `Selected Peaks Information):
    * **Pozycję piku** - zależnie od wybranej opcji z lub bez transformacji
    * **Amplituda (Amplitude)**: Wysokość dopasowanej funkcji Gaussa, reprezentująca maksymalną wartość idealnego piku.
    * **Intensywność (Intensity)**: Całkowita *objętość* piku, czyli całka pod powierzchnią dopasowanej funkcji Gaussa. Obliczana jest na podstawie parametrów funkcji ze wzoru: $I=2\pi\cdot∣A∣\cdot∣\sigma_x∣\cdot∣\sigma_y∣$, gdzie $A$ to amplituda, a $\sigma_x$ i $\sigma_y$ to odchylenia standardowe Gaussa w osiach x i y.
    * **Wartość Maksymalna (Max Value)**: Dla porównania, zapisywana jest również maksymalna wartość piksela znaleziona w surowych danych wewnątrz obszaru ROI.
    * **Obliczenie Odległości Międzypłaszczyznowej (d-spacing)**: Przeliczane są skorygowane współrzędne pikselowe na fizyczne jednostki przestrzeni odwrotnej (`nm⁻¹`), używając do tego danych kalibracyjnych obrazu (`Lx` i `Ly` w nanometrach), które pochodzą z oryginalnego pliku STM. Oblicza długość (moduł) tego wektora w przestrzeni odwrotnej, $∣g^∗∣$. Wynikiem jest odległość międzypłaszczyznowa (`d`) w nanometrach, obliczona ze wzoru $d=\frac{1}{∣g^∗∣}$. Reprezentuje ona rzeczywistą okresowość w sieci krystalicznej, która odpowiada za dany pik dyfrakcyjny.

3. Po kliknięciu przycisku `Calculate Domain Wall Parameters` program obliczy:
* **Odległość między pikami w przestrzeni odwrotnej ($\Delta g^∗$)**: Jest to wektorowa odległość między centrami piku głównego i satelitarnego. Proces obliczeniowy wygląda następująco:
  * **Wektor Różnicy w Pikselach**: Program oblicza wektor łączący centrum piku głównego z centrum piku satelitarnego.
  * **Konwersja na Jednostki Fizyczne**: Następnie, wektor ten jest przeliczany z pikseli na jednostki przestrzeni odwrotnej (`nm⁻¹`), wykorzystując dane kalibracyjne obrazu (`Lx` i `Ly`).
  * **Obliczenie Długości**: Ostateczna wartość $\Delta g^∗$ to długość (moduł) tego wektora w jednostkach `nm⁻¹`. Jest to kluczowa wartość do dalszych obliczeń.
* **Okresowość ściany domenowej (P) w przestrzeni rzeczywistej**: Jest to najważniejszy parametr fizyczny wynikający z tej analizy. Reprezentuje on rzeczywistą, periodyczną odległość między elementami nadstruktury (np. pasami domenowymi) na powierzchni próbki. Program oblicza go bezpośrednio z odległości w przestrzeni odwrotnej, korzystając z fundamentalnej zależności: $P=\frac{1}{\Delta g^∗}$. Wynik podawany jest w nanometrach (nm).
* Stosunek intensywności, amplitudy i maksymalnej wartości piku satelitarnego do głównego.

### 6. Symulacja Obrazów STM/FFT
Narzędzie `STM/FFT Simulation...` (menu `Analysis`) pozwala na stworzenie wirtualnego modelu powierzchni (substrat + adsorbat + nadstruktura), wygenerowanie na jego podstawie symulowanego obrazu STM oraz jego transformaty Fouriera (FFT).

1. Po wybraniu opcji `STM/FFT Simulation...`, otwiera się okno dialogowe z trzema głównymi panelami wizualizacyjnymi:
  * **Simulated STM**: Podgląd generowanego w czasie rzeczywistym obrazu topografii.
  * **Simulated FFT**: Obraz FFT obliczony na podstawie symulowanego obrazu STM.
  * **Experimental FFT**: Obraz FFT z Twoich rzeczywistych danych, wczytany jako punkt odniesienia do porównań.
Po lewej stronie znajduje się panel `Simulation Controls`, gdzie budujesz swój model teoretyczny.

2. Budowa Modelu Teoretycznego
W panelu kontrolnym możesz dostosować wszystkie aspekty symulowanej powierzchni:
* Definicja Sieci (Lattice):
  * **Substrate**: Wybierz substrat, który stanowi podłoże dla Twojego modelu (np. `Au(111)`).
  * **Adsorbate**: Wybierz typ adsorbatu. Opcja `Iodine (predefined)` używa modelu dla tej cząsteczki.
* Definicja Nadstruktury (`Domain & Misfit`):
Tutaj definiujesz fizyczne właściwości nadstruktury, takie jak ściany domenowe. Zmiana tych parametrów na żywo aktualizuje symulowane obrazy, pozwalając na intuicyjne "dopasowywanie" modelu.
  * **Compression**: Symuluje jednoosiową kompresję lub rozciągnięcie sieci adsorbatu względem substratu.
  * **Stripe Width**: Definiuje okresowość (szerokość) pasów w modelu prążkowym (`striped`).
  * **Relax Width**: Określa szerokość obszaru przejściowego, czyli samej ściany domenowej.
  * **Domain Type**: Pozwala wybrać typ ściany domenowej (np. `Heavy`, `Super Heavy`), który w modelu fizycznym odpowiada za konkretne przesunięcie fazowe atomów wewnątrz domeny.
  * **Symmetry**: Umożliwia wybór ogólnej symetrii nadstruktury: `Striped` (pasy w jednym kierunku) lub `Hexagonal`.

3. Analiza i Porównanie Wyników
Gdy już zbudujesz swój model, możesz go porównać z eksperymentem na dwa sposoby:
* **Porównanie Wizualne**: Obserwuj panel `Simulated FFT` i porównuj go z panelem `Experimental FFT`. Dostosowuj parametry (np. `Compression`, `Stripe Width`), aby pozycje, kształty i względne jasności pików w symulacji jak najlepiej odpowiadały tym w eksperymencie.
* **Analiza Ilościowa (Quantitative Analysis)**: Kliknij przycisk `Analyze Simulated Peaks`. Program automatycznie znajdzie pozycje piku głównego i satelitarnego w symulacji i na podstawie dopasowania funkcji Gaussa obliczy dla nich parametry.
W panelu `Automated Analysis` wyświetlone zostaną stosunki intensywności i amplitudy piku satelitarnego do głównego.

4. Zapisanie Wyniku
Jeśli chcesz zachować wygenerowane FFT do dalszej analizy (np. użyć na nim innych narzędzi LFA - możesz użyć `Analyze Domain Walls...` i manualnie wybrać piki do analizy (rekomendowane)), kliknij przycisk `Load Simulation to Main Window`. Spowoduje to dodanie symulowanego obrazu FFT jako nowego elementu w panelu History w głównym oknie aplikacji.

### 7. Rekonstrukcja z Przestrzeni Odwrotnej
Narzędzie `Real Space Reconstruction...` (dostępne w menu `Analysis`) pozwala na wykonanie analizy *odwrotnej* – czyli przejścia z obrazu FFT z powrotem do przestrzeni rzeczywistej. Celem nie jest odtworzenie oryginalnego obrazu STM (co wymagałoby zachowania fazy, którą FFT traci), ale zrozumienie, jakie cechy w przestrzeni rzeczywistej odpowiadają za konkretne elementy widma Fouriera.
Dialog oferuje dwie główne metody analizy: Autokorelację oraz Rekonstrukcję z Maski.

1. Autokorelacja (Funkcja Pattersona)
Autokorelacja to matematyczna operacja, która tworzy mapę wszystkich możliwych wektorów łączących pary punktów w oryginalnym obrazie. W kontekście analizy STM, jasne punkty na mapie autokorelacji odpowiadają najczęściej powtarzającym się wektorom w sieci krystalicznej. W praktyce, wektory łączące centrum mapy z tymi jasnymi punktami to wektory sieci rzeczywistej ($a_1$,$a_2$). 

W oknie `Real Space Reconstruction` wybierz tryb `Calculate Autocorrelation`. Zaznacz opcję `Remove DC Component` jeżeli konieczne jest usunięcie centralnego, najjaśniejszego piku (0,0) z obrazu FFT przed obliczeniami. Pozostawienie go spowodowałoby powstanie bardzo jasnego tła, które zdominowałoby obraz wynikowy.
Wybierz skalę wyświetlania (`Display Scale`), np. `Log Power`, aby lepiej uwidocznić subtelne piki na wynikowej mapie.

W panelu `Reconstructed Real Space` pojawi się mapa autokorelacji. Jest ona zawsze centrosymetryczna. Poszukaj na niej jasnych punktów poza samym centrum. Wektory od centrum obrazu do tych punktów odpowiadają wektorom sieci rzeczywistej.

2. Rekonstrukcja z Maski (ROI lub Spoty)
Ta metoda pozwala na *wycięcie* z widma FFT tylko tych częstotliwości, które Cię interesują, i sprawdzenie, jakie struktury w przestrzeni rzeczywistej one tworzą. Wybór Trybu: Wybierz tryb `Mask with ROIs` (maskowanie prostokątnymi obszarami) lub `Mask with Spots` (maskowanie pojedynczymi pikami).

Tworzenie Maski:
* **Dla Mask with ROIs**: Kliknij `Add ROI`, aby dodać na obrazie FFT prostokątny obszar. Możesz go przesuwać i skalować, aby objąć interesujące Cię piki.
**UWAGA**: Zaznacz opcję `Add Symmetric ROI`. Spowoduje to automatyczne dodanie lustrzanego odbicia każdego ROI względem centrum obrazu.
* **Dla Mask with Spots**:
Klikaj na obrazie FFT, aby umieścić małe ROI do dopasowania. Kliknij `Select Spot`, aby program precyzyjnie zlokalizował pik (metoda `2D Gaussian`) i dodał go do maski w postaci plamy Gaussa. Rozmiar tej plamy (`sigma`) możesz kontrolować za pomocą `spot_mask_size_spinbox`. W środkowym panelu `Mask Preview` na bieżąco zobaczysz, jak wygląda tworzona przez Ciebie maska. Gdy maska jest gotowa, kliknij `Reconstruct from Mask`.