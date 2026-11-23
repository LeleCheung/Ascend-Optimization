param(
    [string]$ReportDirectory = (Join-Path $PSScriptRoot '../../reports/hgkda-flow-slides-20261122'),
    [string[]]$Stems = @('算子优化-基本回路', 'HG-KDA-overview-简版', '模拟器-证据与优化', 'HG-KDA-过渡与流程-三页')
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$reportTask = (Resolve-Path -LiteralPath $ReportDirectory).Path
$pptTask = $null
$deckTask = $null
$initialCountTask = -1
$checksTask = @()
try {
    $pptTask = New-Object -ComObject PowerPoint.Application
    $initialCountTask = $pptTask.Presentations.Count
    foreach ($stemTask in $Stems) {
        $deckTask = $pptTask.Presentations.Open((Join-Path $reportTask ($stemTask + '.pptx')), -1, 0, 0)
        $deckTask.SaveAs((Join-Path $reportTask ($stemTask + '.pdf')), 32)
        $overflowsTask = @()
        foreach ($slideTask in $deckTask.Slides) {
            $suffixTask = if ($deckTask.Slides.Count -eq 1) { '' } else { '-' + $slideTask.SlideIndex }
            $slideTask.Export((Join-Path $reportTask ($stemTask + $suffixTask + '.png')), 'PNG', 2560, 1440)
            foreach ($shapeTask in $slideTask.Shapes) {
                if ($shapeTask.HasTextFrame -eq -1 -and $shapeTask.TextFrame.HasText -eq -1) {
                    $widthTask = $shapeTask.TextFrame2.TextRange.BoundWidth
                    $heightTask = $shapeTask.TextFrame2.TextRange.BoundHeight
                    if ($widthTask -gt ($shapeTask.Width + 1) -or $heightTask -gt ($shapeTask.Height + 1)) {
                        $overflowsTask += @{
                            slide = $slideTask.SlideIndex
                            shape = $shapeTask.Name
                            text = $shapeTask.TextFrame.TextRange.Text
                            text_width = $widthTask
                            text_height = $heightTask
                            box_width = $shapeTask.Width
                            box_height = $shapeTask.Height
                        }
                    }
                }
            }
        }
        $checksTask += @{ file = $stemTask; slides = $deckTask.Slides.Count; text_overflows = @($overflowsTask) }
        $deckTask.Close()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($deckTask)
        $deckTask = $null
    }
    $checksTask | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reportTask 'native-export-check.json') -Encoding UTF8
    $checksTask | ConvertTo-Json -Depth 8
} finally {
    if ($null -ne $deckTask) {
        $deckTask.Close()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($deckTask)
    }
    if ($null -ne $pptTask) {
        if ($initialCountTask -eq 0 -and $pptTask.Presentations.Count -eq 0) { $pptTask.Quit() }
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($pptTask)
    }
}
