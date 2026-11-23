param(
    [string]$MainPresentation = 'E:\0_XSAI\00_KernelAgent\0_HGKDA_docs\BAAI-reports\06_work\智源20261124汇报.pptx'
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$reportTask = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../reports/overview-msprof-animation-v2-20261123'))
$materialTask = Join-Path (Split-Path -Parent $MainPresentation) 'material/01_开场与总览'
New-Item -ItemType Directory -Path $reportTask -Force | Out-Null
New-Item -ItemType Directory -Path $materialTask -Force | Out-Null
$stemTask = 'HG-KDA-overview-msprof-深蓝淡入版'
$workingTask = Join-Path $reportTask ($stemTask + '.pptx')
$hashBeforeTask = (Get-FileHash -LiteralPath $MainPresentation -Algorithm SHA256).Hash
Copy-Item -LiteralPath $MainPresentation -Destination (Join-Path $reportTask 'main-source-snapshot.pptx') -Force
Copy-Item -LiteralPath $MainPresentation -Destination $workingTask -Force

function Get-ShapeByIdTask($slideTask, [int]$idTask) {
    foreach ($shapeTask in $slideTask.Shapes) { if ($shapeTask.Id -eq $idTask) { return $shapeTask } }
    throw "Missing shape ID $idTask"
}

function Add-CommandOverlayTask($slideTask, $boxTask, $labelTask, [string]$commandTask, [string]$nameTask) {
    $overlayTask = $boxTask.Duplicate().Item(1)
    $overlayTask.Name = $nameTask
    $overlayTask.Left = $boxTask.Left - 16
    $overlayTask.Top = $boxTask.Top - 6
    $overlayTask.Width = $boxTask.Width + 32
    $overlayTask.Height = $boxTask.Height + 12
    $overlayTask.Fill.Solid()
    $overlayTask.Fill.ForeColor.RGB = 7159571 # #133F6D, matching the main diagram navy.
    $overlayTask.Line.ForeColor.RGB = 7159571
    $frameTask = $overlayTask.TextFrame
    $frameTask.MarginLeft = $frameTask.MarginRight = 6
    $frameTask.MarginTop = $frameTask.MarginBottom = 0
    $frameTask.AutoSize = 0
    $frameTask.WordWrap = 0
    $frameTask.VerticalAnchor = 3
    $frameTask.TextRange.Text = $commandTask
    $frameTask.TextRange.Font.Name = $labelTask.TextFrame.TextRange.Font.Name
    $frameTask.TextRange.Font.NameFarEast = 'Microsoft YaHei'
    $frameTask.TextRange.Font.Size = 18
    $frameTask.TextRange.Font.Bold = $labelTask.TextFrame.TextRange.Font.Bold
    $frameTask.TextRange.Font.Color.RGB = 16777215
    $frameTask.TextRange.ParagraphFormat.Alignment = 2
    $frameTask.TextRange.ParagraphFormat.SpaceBefore = 0
    $frameTask.TextRange.ParagraphFormat.SpaceAfter = 0
    $overlayTask.Left = $boxTask.Left - 16
    $overlayTask.Top = $boxTask.Top - 6
    $overlayTask.Width = $boxTask.Width + 32
    $overlayTask.Height = $boxTask.Height + 12
    $overlayTask.ZOrder(0)
    $overlayTask.AlternativeText = "Click replacement: $commandTask"
    return $overlayTask
}

$pptTask = $null
$deckTask = $null
try {
    try { $pptTask = [Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application') }
    catch { $pptTask = New-Object -ComObject PowerPoint.Application }
    $openBeforeTask = @(foreach ($openTask in $pptTask.Presentations) { $openTask.FullName })
    $deckTask = $pptTask.Presentations.Open($workingTask, 0, 0, 0)
    $sourceSlideCountTask = $deckTask.Slides.Count
    $sourceSlideTask = $deckTask.Slides.Item(4)
    $sourceTitleTask = (Get-ShapeByIdTask $sourceSlideTask 2).TextFrame.TextRange.Text
    if (-not $sourceTitleTask.Contains('Hardware-Grounded Evidence')) { throw 'Unexpected source overview slide' }
    $sourceEffectsTask = $sourceSlideTask.TimeLine.MainSequence.Count
    if ($sourceEffectsTask -ne 0) { throw 'Source overview already has animations; review before adding click effects' }
    $sourceSlideTask.Export((Join-Path $reportTask 'source-slide4.png'), 'PNG', 2560, 1440)
    for ($indexTask = $deckTask.Slides.Count; $indexTask -ge 1; $indexTask--) {
        if ($indexTask -ne 4) { $deckTask.Slides.Item($indexTask).Delete() }
    }
    $slideTask = $deckTask.Slides.Item(1)
    $slideTask.Export((Join-Path $reportTask 'state-0.png'), 'PNG', 2560, 1440)
    $initialMatchesTask = ((Get-FileHash -LiteralPath (Join-Path $reportTask 'source-slide4.png')).Hash -eq (Get-FileHash -LiteralPath (Join-Path $reportTask 'state-0.png')).Hash)
    if (-not $initialMatchesTask) { throw 'Standalone initial slide changed visually' }
    $profileBoxTask = Get-ShapeByIdTask $slideTask 21
    $profileLabelTask = Get-ShapeByIdTask $slideTask 22
    $simBoxTask = Get-ShapeByIdTask $slideTask 23
    $simLabelTask = Get-ShapeByIdTask $slideTask 24
    if ($profileLabelTask.TextFrame.TextRange.Text -ne '成对 profiling' -or $simLabelTask.TextFrame.TextRange.Text -ne '模拟器分析') { throw 'Unexpected labels' }
    $sequenceTask = $slideTask.TimeLine.MainSequence

    $profileOverlayTask = Add-CommandOverlayTask $slideTask $profileBoxTask $profileLabelTask 'msprof op' '点击1_msprof_op'
    $old1Task = $sequenceTask.AddEffect($profileLabelTask, 10, 0, 1)
    $old1Task.Exit = -1
    $old1Task.Timing.Duration = 0.45
    $old1Task.Timing.TriggerDelayTime = 0
    $effect1Task = $sequenceTask.AddEffect($profileOverlayTask, 10, 0, 2)
    $effect1Task.Timing.Duration = 0.45
    $effect1Task.Timing.TriggerDelayTime = 0
    $slideTask.Export((Join-Path $reportTask 'state-1.png'), 'PNG', 2560, 1440)

    $simOverlayTask = Add-CommandOverlayTask $slideTask $simBoxTask $simLabelTask 'msprof op simulator' '点击2_msprof_op_simulator'
    $old2Task = $sequenceTask.AddEffect($simLabelTask, 10, 0, 1)
    $old2Task.Exit = -1
    $old2Task.Timing.Duration = 0.45
    $old2Task.Timing.TriggerDelayTime = 0
    $effect2Task = $sequenceTask.AddEffect($simOverlayTask, 10, 0, 2)
    $effect2Task.Timing.Duration = 0.45
    $effect2Task.Timing.TriggerDelayTime = 0
    $slideTask.Export((Join-Path $reportTask 'state-2.png'), 'PNG', 2560, 1440)
    $slideTask.SlideShowTransition.AdvanceOnTime = 0
    $slideTask.SlideShowTransition.AdvanceOnClick = -1
    $noteTask = '动画：进入本页显示原总图。第一次单击，“成对 profiling”淡出，同时更大的深蓝白字 msprof op 框淡入覆盖。第二次单击，“模拟器分析”淡出，同时更大的深蓝白字 msprof op simulator 框淡入覆盖。每次过渡0.45秒，不移动。主汇报未修改。本页复制到主汇报时选择保留源格式，动画随幻灯片保留。'
    foreach ($notesShapeTask in $slideTask.NotesPage.Shapes.Placeholders) {
        if ($notesShapeTask.PlaceholderFormat.Type -eq 2) {
            $notesShapeTask.TextFrame.TextRange.Text = $notesShapeTask.TextFrame.TextRange.Text.TrimEnd() + "`r`r" + $noteTask
            break
        }
    }
    $animationsTask = @(foreach ($effectTask in $sequenceTask) {
        @{ shape=$effectTask.Shape.Name; shape_id=$effectTask.Shape.Id; effect_type=[int]$effectTask.EffectType; trigger_type=[int]$effectTask.Timing.TriggerType; duration=$effectTask.Timing.Duration; exit=[bool]$effectTask.Exit }
    })
    $overflowTask = @()
    foreach ($overlayTask in @($profileOverlayTask, $simOverlayTask)) {
        $boundWidthTask = $overlayTask.TextFrame2.TextRange.BoundWidth
        $boundHeightTask = $overlayTask.TextFrame2.TextRange.BoundHeight
        if ($boundWidthTask -gt $overlayTask.Width-12+1 -or $boundHeightTask -gt $overlayTask.Height+1) { $overflowTask += $overlayTask.Name }
    }
    if ($overflowTask.Count -gt 0) { throw ('Overlay text overflow: ' + ($overflowTask -join ', ')) }
    $deckTask.Save()
    $deckTask.Close()
    $deckTask = $null
    Copy-Item -LiteralPath $workingTask -Destination (Join-Path $materialTask ($stemTask + '.pptx')) -Force
    $hashAfterTask = (Get-FileHash -LiteralPath $MainPresentation -Algorithm SHA256).Hash
    $resultTask = @{
        source=$MainPresentation; source_slide=4; source_slides=$sourceSlideCountTask; title=$sourceTitleTask
        source_sha256_before=$hashBeforeTask; source_sha256_after=$hashAfterTask; source_unchanged=($hashBeforeTask -eq $hashAfterTask)
        initial_render_unchanged=$initialMatchesTask; source_animations=$sourceEffectsTask
        output=(Join-Path $materialTask ($stemTask + '.pptx')); animations=$animationsTask; text_overflows=$overflowTask
        open_presentations_before=$openBeforeTask; open_presentations_after=@(foreach ($openTask in $pptTask.Presentations) { $openTask.FullName })
    }
    $resultTask | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $reportTask 'validation.json') -Encoding UTF8
    $resultTask | ConvertTo-Json -Depth 10
    if (-not $resultTask.source_unchanged) { throw 'Source changed while creating standalone copy' }
} finally {
    if ($null -ne $deckTask) { $deckTask.Close() }
    if ($null -ne $pptTask) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($pptTask) }
}
