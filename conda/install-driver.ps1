$ErrorActionPreference = 'Stop'
try {
    $folder = Join-Path $env:PREFIX 'Lib\site-packages\stellarnet_driverLibs\windows_only'
    $process = Start-Process -FilePath (Join-Path $folder 'InstallDriver.exe') `
        -WorkingDirectory $folder -Verb RunAs -WindowStyle Hidden -Wait -PassThru
    $process.WaitForExit()
    $process.Refresh()
    if ($null -eq $process.ExitCode) { throw 'The driver installer returned no status.' }
    $code = [long]$process.ExitCode
    # DPInst: bit 31 and bits 16-23 indicate failure; low bytes count drivers.
    # Bit 30 requests a reboot. See Microsoft's "DPInst Return Code" reference.
    if ($code -band 0x80FF0000L) {
        throw ('Driver installation failed (status 0x{0:X8}).' -f ($code -band 0xFFFFFFFFL))
    }
    if ($code -band 0x40000000L) {
        Add-Content -LiteralPath (Join-Path $env:PREFIX '.messages.txt') `
            -Value 'StellarNet USB driver installed; restart Windows to finish setup.'
    }
    exit 0
} catch {
    Write-Error -Message $_ -ErrorAction Continue
    exit 1
}
