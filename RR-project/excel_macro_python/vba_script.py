Option Explicit

' ========= USER SETTINGS (edit these two) =========
Private Const PYTHON_EXE As String = "python"   ' e.g. "C:\Users\you\Miniconda3\python.exe"
Private Const PIPELINE_SCRIPT As String = "C:\path\to\your\pipeline_cli.py" ' your Python entrypoint
' ==================================================

Public Sub Run_RentRoll_Pipeline()
    Dim rrSheetName As String: rrSheetName = "RAW_RENT_ROLL"
    Dim concSheetName As String: concSheetName = "CONCESSIONS" ' optional; will be skipped if missing
    Dim outputSheetName As String: outputSheetName = "RENT_ROLL_OUTPUT"
    Dim logSheetName As String: logSheetName = "PIPELINE_LOG"
    
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
    Dim tempDir As String: tempDir = Environ$("TEMP") & "\rr_pipeline_" & Format(Now, "yyyymmdd_hhnnss")
    Dim rrCsv As String: rrCsv = tempDir & "\raw_rent_roll.csv"
    Dim concCsv As String: concCsv = tempDir & "\concessions.csv"
    Dim outCsv As String: outCsv = tempDir & "\output.csv"
    Dim ok As Boolean
    
    On Error GoTo FAIL_FAST
    CreateFolderIfMissing tempDir
    
    ' Validate required sheet
    If Not SheetExists(rrSheetName) Then
        Raise "Required sheet '" & rrSheetName & "' not found."
    End If
    
    ' Export sheets to CSVs
    ExportSheetToCSV ThisWorkbook.Worksheets(rrSheetName), rrCsv
    
    Dim hasConcessions As Boolean
    hasConcessions = SheetExists(concSheetName)
    If hasConcessions Then
        ExportSheetToCSV ThisWorkbook.Worksheets(concSheetName), concCsv
    Else
        concCsv = "" ' signal missing
    End If
    
    ' Run Python pipeline
    Dim stdout As String, stderr As String, exitCode As Long
    RunPythonPipeline PYTHON_EXE, PIPELINE_SCRIPT, rrCsv, concCsv, outCsv, stdout, stderr, exitCode
    
    ' Log output for traceability
    WriteLogSheet logSheetName, stdout, stderr, exitCode, rrCsv, concCsv, outCsv
    
    If exitCode <> 0 Then
        Raise "Python pipeline exited with code " & exitCode & ". See '" & logSheetName & "' for details."
    End If
    
    If Not fso.FileExists(outCsv) Then
        Raise "Pipeline finished but no output file found at: " & outCsv & ". See '" & logSheetName & "'."
    End If
    
    ' Import result CSV into Excel
    ImportCSVToWorksheet outCsv, outputSheetName
    
    ' Cleanup temp files (optional; comment out to keep artifacts)
    SafeDeleteFolder tempDir
    
    MsgBox "Done. Results written to '" & outputSheetName & "'.", vbInformation
    Exit Sub

FAIL_FAST:
    On Error Resume Next
    WriteLogSheet logSheetName, stdout, stderr, exitCode, rrCsv, concCsv, outCsv
    On Error GoTo 0
    MsgBox "Failed: " & Err.Description, vbCritical
End Sub


' ---------- Helpers ----------

Private Sub ExportSheetToCSV(ws As Worksheet, destCsv As String)
    Dim tmpWB As Workbook
    Application.DisplayAlerts = False
    ws.Copy ' copies sheet into a new workbook as ActiveWorkbook
    Set tmpWB = ActiveWorkbook
    tmpWB.SaveAs Filename:=destCsv, FileFormat:=xlCSVUTF8, CreateBackup:=False
    tmpWB.Close SaveChanges:=False
    Application.DisplayAlerts = True
End Sub

Private Sub RunPythonPipeline( _
    ByVal pyExe As String, _
    ByVal scriptPath As String, _
    ByVal rrCsv As String, _
    ByVal concCsv As String, _
    ByVal outCsv As String, _
    ByRef stdout As String, _
    ByRef stderr As String, _
    ByRef exitCode As Long)
    
    Dim shell As Object: Set shell = CreateObject("WScript.Shell")
    Dim cmd As String
    
    ' Build command line; adjust flags to match your CLI
    ' Expected Python CLI: pipeline_cli.py --rent-roll <csv> [--concessions <csv>] --out <csv>
    cmd = """" & pyExe & """ """ & scriptPath & """ --rent-roll """ & rrCsv & """ --out """ & outCsv & """"
    If Len(concCsv) > 0 Then
        cmd = cmd & " --concessions " & """" & concCsv & """"
    End If
    
    Dim execObj As Object
    Set execObj = shell.Exec(cmd)
    
    ' Wait and capture streams
    Dim line As String
    stdout = ""
    stderr = ""
    
    Do While execObj.Status = 0
        DoEvents
        ' Drain as it comes
        If Not execObj.StdOut.AtEndOfStream Then
            line = execObj.StdOut.ReadAll
            If Len(line) > 0 Then stdout = stdout & line
        End If
        If Not execObj.StdErr.AtEndOfStream Then
            line = execObj.StdErr.ReadAll
            If Len(line) > 0 Then stderr = stderr & line
        End If
    Loop
    
    ' Final drain
    If Not execObj.StdOut.AtEndOfStream Then stdout = stdout & execObj.StdOut.ReadAll
    If Not execObj.StdErr.AtEndOfStream Then stderr = stderr & execObj.StdErr.ReadAll
    
    exitCode = execObj.ExitCode
End Sub

Private Sub ImportCSVToWorksheet(csvPath As String, targetSheetName As String)
    Dim ws As Worksheet
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False
    
    If SheetExists(targetSheetName) Then
        Set ws = ThisWorkbook.Worksheets(targetSheetName)
        ws.Cells.Clear
    Else
        Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        ws.Name = targetSheetName
    End If
    
    ' Use QueryTable for robust CSV import (handles quoted commas)
    Dim qt As QueryTable
    Set qt = ws.QueryTables.Add(Connection:="TEXT;" & csvPath, Destination:=ws.Range("A1"))
    With qt
        .TextFilePromptOnRefresh = False
        .TextFilePlatform = 65001          ' UTF-8
        .TextFileCommaDelimiter = True
        .TextFileOtherDelimiter = False
        .TextFileConsecutiveDelimiter = False
        .TextFileTrailingMinusNumbers = True
        .TextFileColumnDataTypes = Array(1) ' all text -> Excel infers later if needed
        .AdjustColumnWidth = True
        .PreserveColumnInfo = True
        .Refresh BackgroundQuery:=False
        .Delete
    End With
    
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
End Sub

Private Sub WriteLogSheet(logSheetName As String, _
    ByVal stdout As String, ByVal stderr As String, ByVal exitCode As Long, _
    ByVal rrCsv As String, ByVal concCsv As String, ByVal outCsv As String)
    
    Dim ws As Worksheet
    If SheetExists(logSheetName) Then
        Set ws = ThisWorkbook.Worksheets(logSheetName)
        ws.Cells.Clear
    Else
        Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        ws.Name = logSheetName
    End If
    
    ws.Range("A1").Value = "Timestamp"
    ws.Range("B1").Value = Format(Now, "yyyy-mm-dd hh:nn:ss")
    ws.Range("A3").Value = "ExitCode"
    ws.Range("B3").Value = exitCode
    ws.Range("A5").Value = "RR CSV"
    ws.Range("B5").Value = rrCsv
    ws.Range("A6").Value = "Concessions CSV"
    ws.Range("B6").Value = IIf(Len(concCsv) > 0, concCsv, "(none)")
    ws.Range("A7").Value = "Output CSV"
    ws.Range("B7").Value = outCsv
    ws.Range("A9").Value = "STDOUT"
    ws.Range("A10").Value = stdout
    ws.Range("A10").EntireRow.RowHeight = 15
    ws.Range("A10").EntireColumn.ColumnWidth = 120
    ws.Range("A10").WrapText = True
    ws.Range("A10").EntireRow.AutoFit
    
    ws.Range("A12").Value = "STDERR"
    ws.Range("A13").Value = stderr
    ws.Range("A13").EntireColumn.ColumnWidth = 120
    ws.Range("A13").WrapText = True
    ws.Range("A13").EntireRow.AutoFit
End Sub

Private Function SheetExists(shtName As String) As Boolean
    Dim sht As Worksheet
    On Error Resume Next
    Set sht = ThisWorkbook.Worksheets(shtName)
    SheetExists = Not sht Is Nothing
    On Error GoTo 0
End Function

Private Sub CreateFolderIfMissing(path As String)
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
    If Not fso.FolderExists(path) Then fso.CreateFolder path
End Sub

Private Sub SafeDeleteFolder(path As String)
    On Error Resume Next
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
    If fso.FolderExists(path) Then fso.DeleteFolder path, True
    On Error GoTo 0
End Sub

Private Sub Raise(msg As String)
    Err.Raise vbObjectError + 513, "RentRollPipeline", msg
End Sub
