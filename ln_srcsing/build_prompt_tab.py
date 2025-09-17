Option Explicit

Public Sub BuildPromptsTab()
    Dim ws As Worksheet
    Dim lo As ListObject
    Dim rng As Range
    Dim headers As Variant
    Dim data As Variant
    Dim i As Long
    
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False
    
    '--- Create or clear the "prompts" sheet
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets("prompts")
    On Error GoTo 0
    
    If ws Is Nothing Then
        Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        ws.Name = "prompts"
    Else
        ws.Cells.Clear
    End If
    
    '--- Headers (kept "Promot" as requested)
    headers = Array("Variable_names", "Promot", "instructions")
    ws.Range("A1").Resize(1, UBound(headers) + 1).Value = headers
    
    '--- Seed ~half dozen variables
    '   Edit these as you like; keep "Promot" concise and put longer guidance in "instructions".
    data = Array( _
        Array("LoanAmount", _
              "Extract the loan amount from the workbook.", _
              "Return only a numeric value with no currency symbol; if multiple amounts exist, choose the primary committed amount."), _
        Array("InterestRate", _
              "Extract the stated interest rate.", _
              "Return a percentage as a decimal (e.g., 0.0625 for 6.25%). If a range exists, choose the initial/base rate."), _
        Array("MaturityDate", _
              "Extract the loan maturity date.", _
              "Return in ISO format YYYY-MM-DD. If multiple tranches, pick the senior tranche maturity."), _
        Array("CollateralType", _
              "Identify the collateral type.", _
              "Choose from: RealEstate, Securities, Cash, Equipment, Other. If unclear, write 'Other' and add a short note."), _
        Array("BorrowerName", _
              "Extract the legal borrower name.", _
              "Prefer the exact legal entity as shown in term sheet/cover page; no abbreviations."), _
        Array("OriginationFees", _
              "Extract total upfront fees at origination.", _
              "Return numeric sum in dollars; include arrangement/underwriting fees and OID if specified.") _
    )
    
    '--- Write data
    Set rng = ws.Range("A2").Resize(UBound(data) + 1, 3)
    For i = LBound(data) To UBound(data)
        rng.Rows(i - LBound(data) + 1).Columns(1).Value = data(i)(0)
        rng.Rows(i - LBound(data) + 1).Columns(2).Value = data(i)(1)
        rng.Rows(i - LBound(data) + 1).Columns(3).Value = data(i)(2)
    Next i
    
    '--- Convert to a Table for nicer filtering/sorting
    Set lo = ws.ListObjects.Add(xlSrcRange, ws.Range("A1").CurrentRegion, , xlYes)
    lo.Name = "tblPrompts"
    lo.TableStyle = "TableStyleMedium2"
    
    '--- Column widths & wrapping
    ws.Columns("A").ColumnWidth = 22
    ws.Columns("B").ColumnWidth = 60
    ws.Columns("C").ColumnWidth = 52
    ws.Columns("B:C").WrapText = True
    ws.Rows.RowHeight = 18
    
    '--- Freeze header row
    With ws
        .Activate
        .Range("A2").Select
        ActiveWindow.FreezePanes = True
    End With
    
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
End Sub
