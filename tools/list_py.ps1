Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object ProcessId, @{N='Cmd';E={ if ($_.CommandLine) { $_.CommandLine.Substring(0, [Math]::Min(140, $_.CommandLine.Length)) } }} |
  Format-List
