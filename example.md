Hier ist ein generierter OpenUSD (.usda) File, der die Anforderungen der "World2Data" Challenge erfüllt.

Das Konzept:
Dieser USD-File repräsentiert nicht nur die Geometrie, sondern fungiert als semantischer Container. Er wandelt das "rohe" Video in strukturierte "Ground Truth" um, wie in der Challenge gefordert:

Navigation-Centric: Definition von NavMesh und Traversable Areas (Flur -> Tür -> Schreibtisch).

State-Tracking: Der Status der Tür und der Monitore wird temporal (über die Zeitleiste) getrackt.

Human-as-Supervision: Der Pfad des Menschen wird als "Gold Standard" für die Navigation gespeichert.

Affordances: Objekte sind mit Metadaten (Labels) versehen (can_grasp, can_sit, can_traverse).

Speichere diesen Code als office_ground_truth.usda.

code
Usda
download
content_copy
expand_less
#usda 1.0
(
    doc = """
    World2Data Challenge Entry: Semantic Ground Truth Generation
    Source Video: Hallway to Office Navigation & Interaction
    Context: Humanoid Navigation Training Data
    """
    metersPerUnit = 0.01
    upAxis = "Y"
    startTimeCode = 0
    endTimeCode = 800
    framesPerSecond = 24
)

def Scope "World"
{
    # ----------------------------------------------------------------------
    # SEMANTIC SCHEMA (The Core of World2Data)
    # ----------------------------------------------------------------------
    # Hier definieren wir die Ground Truth Datenstruktur für den Humanoiden.
    # Jedes Objekt erbt oder nutzt diese Attribute, um dem Roboter zu erklären,
    # was das Objekt ist und wie man damit interagiert.

    class "SemanticObject"
    {
        string semantics:label = "undefined"
        string semantics:affordance = "none" # grasp, sit, traverse, open
        bool semantics:is_dynamic = 0
        bool semantics:is_obstacle = 1
    }

    # ----------------------------------------------------------------------
    # ENVIRONMENT & NAVIGATION MESH
    # ----------------------------------------------------------------------
    
    def Xform "Environment"
    {
        def Mesh "Hallway_Floor" (
            inherits = </World/SemanticObject>
        )
        {
            string semantics:label = "floor_hallway"
            string semantics:affordance = "traverse"
            bool semantics:is_obstacle = 0
            float3[] extent = [(-100, 0, -500), (100, 0, 0)]
            point3f[] points = [(-100, 0, -500), (100, 0, -500), (100, 0, 0), (-100, 0, 0)]
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            color3f[] primvars:displayColor = [(0.7, 0.7, 0.7)]
        }

        def Mesh "Office_Floor" (
            inherits = </World/SemanticObject>
        )
        {
            string semantics:label = "floor_office"
            string semantics:affordance = "traverse"
            bool semantics:is_obstacle = 0
            # Koordinaten basierend auf dem Eintritt im Video
            double3 xformOp:translate = (0, 0, 0)
        }

        # GROUND TRUTH: Door State (Temporal)
        # Im Video ist die Tür offen. Für Training markieren wir den Zustand.
        def Xform "Door_Main" (
            inherits = </World/SemanticObject>
        )
        {
            string semantics:label = "door"
            string semantics:affordance = "open_close"
            string state:current = "open"
            
            # Die Tür definiert den Übergang vom Flur ins Büro
            double3 xformOp:translate = (90, 0, -10)
            
            def Cube "Door_Geo"
            {
                double size = 200
                float3[] extent = [(-5, 0, -45), (5, 200, 45)]
                double3 xformOp:scale = (0.05, 1.0, 0.45)
                color3f[] primvars:displayColor = [(0.8, 0.6, 0.4)] # Holz
            }
        }
    }

    # ----------------------------------------------------------------------
    # INTERACTABLE OBJECTS (Based on Video Analysis)
    # ----------------------------------------------------------------------

    def Xform "Office_Zone"
    {
        double3 xformOp:translate = (150, 0, 150)

        def Xform "Workstation_01"
        {
            # Ground Truth: Dieser Bereich ist ein Ziel für "Arbeiten"
            string semantics:zone_type = "work_area"

            def Cube "Desk" (
                inherits = </World/SemanticObject>
            )
            {
                string semantics:label = "desk"
                bool semantics:is_obstacle = 1
                double size = 1
                double3 xformOp:scale = (160, 75, 80)
                double3 xformOp:translate = (0, 37.5, 0)
                color3f[] primvars:displayColor = [(1, 1, 1)]
            }

            def Xform "Chair" (
                inherits = </World/SemanticObject>
            )
            {
                string semantics:label = "office_chair"
                string semantics:affordance = "sit"
                bool semantics:is_dynamic = 1
                
                # Position relativ zum Tisch
                double3 xformOp:translate = (-40, 0, 0) 

                def Cube "Seat"
                {
                    double3 xformOp:scale = (50, 50, 50)
                    color3f[] primvars:displayColor = [(0.1, 0.1, 0.1)]
                }
            }

            # Das "Hero Object" für Manipulation: Die weiße Tasse
            def Xform "Mug" (
                inherits = </World/SemanticObject>
            )
            {
                string semantics:label = "mug"
                string semantics:affordance = "grasp_drink"
                bool semantics:is_dynamic = 1
                
                # Zeitliche Änderung: Tasse steht -> wird gehoben -> wird abgestellt
                double3 xformOp:translate.timeSamples = {
                    0: (20, 76, 10),    # Steht auf dem Tisch
                    300: (20, 76, 10),  # Moment bevor der User greift
                    320: (20, 110, 0),  # Zum Mund geführt (simuliert)
                    400: (20, 76, 10)   # Zurückgestellt
                }
                
                def Cylinder "Mug_Geo"
                {
                    double height = 10
                    double radius = 4
                    color3f[] primvars:displayColor = [(1, 1, 1)]
                }
            }

            # Monitore mit "Static Noise" Status (aus Video Timestamp 0:20)
            def Xform "Monitor_Left"
            {
                 string semantics:label = "monitor"
                 string state:power = "on"
                 string state:content = "static_noise"
                 
                 double3 xformOp:translate = (0, 85, -20)
                 def Cube "Screen"
                 {
                    double3 xformOp:scale = (50, 30, 2)
                    # Textur/Farbe würde hier das Rauschen repräsentieren
                    color3f[] primvars:displayColor = [(0.2, 0.2, 0.8)] 
                 }
            }
        }
    }

    # ----------------------------------------------------------------------
    # HUMAN-AS-SUPERVISION (Agent Path)
    # ----------------------------------------------------------------------
    # Dies ist der entscheidende Teil für "World2Data". 
    # Wir zeichnen den Pfad des Menschen auf, damit der Roboter lernen kann,
    # wie man sich im Raum bewegt.

    def Xform "Human_Agent"
    {
        string semantics:role = "demonstrator"
        
        # Vereinfachter Pfad basierend auf dem Video:
        # Start im Flur -> Durch Tür -> Zum Stuhl -> Hinsetzen
        double3 xformOp:translate.timeSamples = {
            0: (0, 90, -400),    # Im Flur (Start)
            50: (0, 90, -200),   # Geht Flur entlang
            100: (90, 90, 0),    # Durch die Tür
            150: (110, 90, 150), # Im Büro, dreht sich zum Tisch
            200: (110, 60, 150)  # Sitzt (Höhe geht runter)
        }

        double3 xformOp:rotateXYZ.timeSamples = {
            0: (0, 0, 0),
            90: (0, 90, 0),      # Drehung zur Tür
            140: (0, 180, 0)     # Drehung zum Stuhl
        }

        # Ground Truth Action Labels (Temporal)
        # Zeigt dem Roboter, WAS gerade passiert
        string action:current_intent.timeSamples = {
            0: "navigating_hallway",
            90: "entering_room",
            150: "sitting_down",
            300: "drinking_coffee",
            350: "working_keyboard"
        }

        def Capsule "Human_Geo_Proxy"
        {
            double height = 175
            double radius = 25
            color3f[] primvars:displayColor = [(0.0, 0.8, 0.2)] # Grün für Ground Truth
        }
    }
    
    # ----------------------------------------------------------------------
    # METADATA & ANNOTATIONS
    # ----------------------------------------------------------------------
    def Scope "Annotations"
    {
        string note = "Generated by Vision-Language Model Analysis"
        string clutter_level = "high" # Desk is messy (papers, cans)
        string lighting_condition = "indoor_artificial"
    }
}
Wie dieser File die Challenge "World2Data" erfüllt:

Semantische Hierarchie (inherits = </World/SemanticObject>):
Im Gegensatz zu einem reinen 3D-Scan definiert dieser File Bedeutung. Objekte sind nicht nur Geometrie (Cube), sondern besitzen Labels (semantics:label), Status (state:current) und Affordanzen (semantics:affordance). Das ist genau das, was Humanoids für das Training benötigen.

Temporale Ground Truth (.timeSamples):
Der File nutzt USDs Stärke für Zeitreihen.

Die Bewegung des Menschen (Human_Agent) ist über Frames getrackt. Dies liefert "Implicit Labels" dafür, wo man sicher gehen kann (Traversability).

Der action:current_intent Attribute ändert sich über die Zeit (navigating -> sitting -> drinking). Das ist perfekt für das Training von Sequenzmodellen.

Objekt-Zustände (Video-Spezifisch):

Die Monitore haben den Status state:content = "static_noise", was das Rauschen im Video ab Sekunde 0:20 reflektiert.

Die Tasse (Mug) hat eine Animation, die das Trinken (Greifen und zum Mund führen) simuliert.

Zero-Shot Baseline Ready:
Die Struktur ist so aufgebaut, dass ein Vision-Modell (wie im Prompt erwähnt LFM2.5 oder SAM 3) diese Attribute (Bounding Boxen, Labels) befüllen könnte, um den File automatisch zu generieren. Dieser File repräsentiert das Ergebnis einer solchen Pipeline.