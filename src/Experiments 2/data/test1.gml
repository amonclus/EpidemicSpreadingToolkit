graph [
  directed 0

  # --- HOUSE BASE ---
  node [ id 0 label "A" ]
  node [ id 1 label "B" ]
  node [ id 2 label "C" ]
  node [ id 3 label "D" ]
  node [ id 4 label "E" ]

  edge [ source 0 target 1 ]
  edge [ source 1 target 2 ]
  edge [ source 2 target 3 ]
  edge [ source 3 target 0 ]
  edge [ source 0 target 2 ]
  edge [ source 1 target 3 ]
  edge [ source 0 target 4 ]
  edge [ source 2 target 4 ]

  # --- ROOF ---
  node [ id 5 label "Roof" ]
  edge [ source 1 target 5 ]
  edge [ source 3 target 5 ]

  # --- TOWER ---
  node [ id 6 label "T1" ]
  node [ id 7 label "T2" ]
  node [ id 8 label "T3" ]
  node [ id 9 label "T4" ]
  node [ id 10 label "T5" ]

  edge [ source 5 target 6 ]
  edge [ source 6 target 7 ]
  edge [ source 7 target 8 ]
  edge [ source 8 target 9 ]
  edge [ source 9 target 10 ]

  # --- BRIDGE ---
  node [ id 11 label "Bridge" ]
  edge [ source 4 target 11 ]
  edge [ source 10 target 11 ]

  # --- SECOND CLUSTER ---
  node [ id 12 label "C1" ]
  node [ id 13 label "C2" ]
  node [ id 14 label "C3" ]
  node [ id 15 label "Tail1" ]
  node [ id 16 label "Tail2" ]
  node [ id 17 label "Tail3" ]

  edge [ source 11 target 12 ]
  edge [ source 12 target 13 ]
  edge [ source 13 target 11 ]

  edge [ source 12 target 14 ]
  edge [ source 13 target 14 ]

  edge [ source 14 target 15 ]
  edge [ source 15 target 16 ]
  edge [ source 16 target 17 ]
]