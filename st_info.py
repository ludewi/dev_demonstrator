"""Information"""
import streamlit as st


def app():
    st.title("Informationsseite")
    st.subheader("Hier findest du grundlegende Informationen zu unserem Demonstrator.")


    """ In diesem Container wird die gesamte Funktionsweise des Demonstators erklärt"""
    example = st.container()
    example.subheader('Was ist der MNIST Datensatz?')
    example.markdown("MNIST ist eine große Datenbank mit handgeschriebenen Ziffern und wird auch häufig für das Training"
                     "und Testen im Bereich des maschinellen Lernens verwendet."
                     "Damit die Zahlen für das maschinelle Lernen verwendet werden können müssen die Zahlen-Bilder des MNIST Datensatz zuerst"
                     "in eine für den Computerverständliche Form gebracht werden."
                     "Der Computer kann nur Zahlen lesen, deswegen transformieren wir das Bild in eine Zahlen representation "
                     "Alle weißen Farbpixel bekommen den Wert 255. Alle schwarzen Farbpixel bekommen den Wert 0. "
                   "Bei der Verwendung des Demonstrators später wirst solch eine Umformung durchführen.")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.empty()
    with col2:
        st.image("pictures/MNIST4.png")
    with col3:
        st.empty()
    """ In diesem Container wird die gesamte Funktionsweise des Demonstators erklärt"""
    explain = st.container()
    explain.subheader('Was ist föderiertes Lernen?')
    col1, col2 = st.columns(2)
    with col1:
        st.image("pictures/FL_2.png")
    with col2:
        #st.image("pictures/Aufbau_FL.png")
        video_file = open('pictures/fl_google.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes, start_time=22)
    
    explain.markdown("Herkömmliche Maschine Learning (ML) Algorithmen werden in der Regel auf Daten trainiert, die von "
                     "verschiedenen Endgeräten wie Mobiltelefonen, Laptops usw. gesammelt und auf einem zentralen Server "
                     "zusammengeführt werden. Die ML- Algorithmen werden dann auf diesen Daten trainiert, um so eine "
                     "Prognose auf neuen Daten zu erstellen. Bei den herkömmlichen ML-Methoden werden auch sensible "
                     "Nutzerdaten an die Server gesendet, wie Chat-Nachrichten oder Bilder. Um dies zu verhindern wurde "
                     "2016 das Konzept des Föderierten Lernens (FL) (engl. federated learning) eingeführt. "
                     "Beim **Föderierten Lernen** können ML-Algorithmen auf lokalen Daten trainiert werden, ohne dass "
                     "Daten mit einer zentralen Instanz ausgetauscht werden. Dies ist ein enormer Vorteil in Sachen Datenschutz.")
    explain.markdown("Ein „Federated Learning System“ (FLS) besteht aus mehreren Clients und einem Server. Unter den Clients "
                     "werden meist Endgeräte wie z.B. Mobiltelefone verstanden, welche die lokalen Daten erzeugen. Der "
                     "Server orchestriert das Lernen. Er gibt vor, wie viele Runden trainiert werden, wie viele Clients "
                     "am Training teilnehmen sollen, welche Hyperparameter verwendet werden sollen usw.. Das Föderierte "
                     "Training findet dann immer nach den gleichen Schritten statt.")
    explain.markdown("1.	Zuerst laden alle Clients das aktuelle ML-Modell (wie z.B. Logistische Regression, CNN, FFNN) "
                     "vom Server herunter.")
    explain.markdown( "2.	Die Clients trainieren das Modell mit ihren lokalen Daten. Dies führt dazu, dass die Parameter"
                     "(meist die Gewichte eines neuronalen Netztes) angepasst werden. Da jeder Client unterschiedliche Daten"
                      "hat, werden auch die Modelle aller Clients nach dem Training unterschiedliche Parameter haben. ")
    explain.markdown("3.	Ist das Training der Clients auf den lokalen Daten abgeschossen, senden sie nur die "
                     "angepassten Parameter an den Server zurück.")
    explain.markdown("4.	Der Server aggregiert alle erhaltenen Parameter zu einem neuen angepassten Modell.")
    explain.markdown("Die Schritte 1-4 werden mehrmals ausgeführt. So ist es möglich ein globales ML-Modell zu trainieren,"
                     "ohne dass die lokalen Daten mit einer zentralen Instanz ausgetauscht werden müssen. "
                     "(vgl. McMahan et al., 2017)")





    #explain.image("./Funktion.gif")

    """ In diesem Container wird der gesamte Aufbau des Demonstators erklärt"""
    setup = st.container()
    setup.subheader('Aufbau des Demonstrators')

    setup.markdown(
        "Der Demonstrator soll dir die funktionsweise des föderrierten Lernen an einem praktischen Beispiel näher bringen."
        "Den Computer den du gerade benutzt fungiert als ein Client. Auf diesem Clients können Daten erzeugt werden "
        "(hier Zahlen zwischen 0-9). Unser Ziel ist ein KI-Modell zu erzeugen, das aufgrundlage unserer erzeugten Daten später "
        "die Zahlen 0-9 klassifizieren kann. Wir wollen aber das keine zentrale Instanz unsere erzeugten Zahlen einsehen kann und "
        "weil ein Client nur sehr wenige Zahlen erzeugen kann wäre es hilfreich die Zahlen die andere Clients erzeugen für unser Modell "
        "verwenden zu können ohne das wir die Zahlen des anderen Clients anschauen können."
        "Das föderrierte Training kann genau das leisten."
        ""
        ""
        "Wenn du der Meinung bist genügend Zahlen erzeugt zu haben, kannst du auf den Knopf **Training starten** klicken."
        "Dann Verbindet sich dein Client mit dem Server für föderriertes Lernen"
        "Der Demonstrator zeigt die dann ich echtzeit was er gerade macht und welche zwischen Ergebnisse erzeugt wurden. "
        "Viel Spass bei verwenden")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.empty()
    with col2:
        st.image("pictures/Aufbau.png")
    with col3:
        st.empty()
    #setup.image("./Aufbau_FL.png")

    #""" In diesem Container werden die Vorteile des Demonstators erklärt"""
    #advantage = st.container()
    #advantage.subheader('Welche Vorteile hat FL?')
    #advantage.markdown("**Datensicherheit**")
    #advantage.markdown("**Größere Datenbasis**")
