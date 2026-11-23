param(
    [string]$MainPresentation = 'E:\0_XSAI\00_KernelAgent\0_HGKDA_docs\BAAI-reports\06_work\智源20261124汇报.pptx'
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$reportLoop = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../reports/main-ppt-loop-20261122'))
$deliveryLoop = Split-Path -Parent $MainPresentation
$backupLoop = Join-Path $deliveryLoop 'material/archive/20261122_主PPT第二页动画调整前'
New-Item -ItemType Directory -Path $reportLoop -Force | Out-Null
New-Item -ItemType Directory -Path $backupLoop -Force | Out-Null
$stampLoop = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, 'China Standard Time').ToString('yyyyMMdd_HHmmss')
Copy-Item -LiteralPath $MainPresentation -Destination (Join-Path $backupLoop ($stampLoop + '_磁盘原件.pptx'))

function Get-PptColorLoop([string]$hexLoop) {
    return [Convert]::ToInt32($hexLoop.Substring(0,2),16) + 256 * [Convert]::ToInt32($hexLoop.Substring(2,2),16) + 65536 * [Convert]::ToInt32($hexLoop.Substring(4,2),16)
}

function Set-BoxLoop($shapeLoop, [double]$xLoop, [double]$yLoop, [double]$wLoop, [double]$hLoop) {
    $shapeLoop.Left = [single]($xLoop * 72)
    $shapeLoop.Top = [single]($yLoop * 72)
    $shapeLoop.Width = [single]($wLoop * 72)
    $shapeLoop.Height = [single]($hLoop * 72)
}

function New-TextLoop($slideLoop, [string]$nameLoop, [string]$textLoop, [double]$xLoop, [double]$yLoop, [double]$wLoop, [double]$hLoop, [single]$sizeLoop, [int]$colorLoop) {
    $shapeLoop = $slideLoop.Shapes.AddTextbox(1, [single]($xLoop*72), [single]($yLoop*72), [single]($wLoop*72), [single]($hLoop*72))
    $shapeLoop.Name = $nameLoop
    $frameLoop = $shapeLoop.TextFrame
    $frameLoop.MarginLeft = $frameLoop.MarginRight = 0
    $frameLoop.MarginTop = $frameLoop.MarginBottom = 0
    $frameLoop.WordWrap = 0
    $frameLoop.AutoSize = 0
    $frameLoop.TextRange.Text = $textLoop
    $frameLoop.TextRange.Font.Name = 'Microsoft YaHei'
    $frameLoop.TextRange.Font.NameFarEast = 'Microsoft YaHei'
    $frameLoop.TextRange.Font.Size = $sizeLoop
    $frameLoop.TextRange.Font.Bold = 0
    $frameLoop.TextRange.Font.Color.RGB = $colorLoop
    $frameLoop.TextRange.ParagraphFormat.Alignment = 2
    Set-BoxLoop $shapeLoop $xLoop $yLoop $wLoop $hLoop
    return $shapeLoop
}

try { $pptLoop = [Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application') }
catch { $pptLoop = New-Object -ComObject PowerPoint.Application }
$deckLoop = $null
foreach ($openLoop in $pptLoop.Presentations) {
    if ($openLoop.FullName -eq $MainPresentation) { $deckLoop = $openLoop; break }
}
if ($null -eq $deckLoop) { $deckLoop = $pptLoop.Presentations.Open($MainPresentation, 0, 0, -1) }
$deckLoop.SaveCopyAs((Join-Path $backupLoop ($stampLoop + '_编辑前快照.pptx')), 24)
$slideLoop = $deckLoop.Slides.Item(2)
if ($slideLoop.Shapes.Item('当前算子版本').TextFrame.TextRange.Text -ne '当前算子版本') { throw 'Unexpected slide 2 layout' }
foreach ($otherLoop in $deckLoop.Slides) {
    if ($otherLoop.SlideIndex -ne 2) { $otherLoop.Export((Join-Path $reportLoop ('before-slide-' + $otherLoop.SlideIndex + '.png')), 'PNG', 1600, 900) }
}
$slideLoop.Export((Join-Path $reportLoop 'slide2-before.png'), 'PNG', 2560, 1440)
$navyLoop = Get-PptColorLoop '133F6D'
$mutedLoop = Get-PptColorLoop '606870'

# Keep the user's title and editable table. Move the loop into the lower-left.
$titleLoop = $slideLoop.Shapes.Item(1)
Set-BoxLoop $titleLoop .55 .48 6.6 .65
$currentLoop = $slideLoop.Shapes.Item('当前算子版本')
$feedbackLoop = $slideLoop.Shapes.Item('执行与反馈')
Set-BoxLoop $currentLoop .80 4.10 2.60 1.00
Set-BoxLoop $feedbackLoop 6.60 4.10 2.60 1.00
$currentLoop.TextFrame.TextRange.Font.Size = 23
$feedbackLoop.TextFrame.TextRange.Font.Size = 23
$connectorLoop = $slideLoop.Shapes.Item('流程连接')
Set-BoxLoop $connectorLoop 3.40 4.60 3.20 0
$returnLinesLoop = @()
foreach ($shapeLoop in $slideLoop.Shapes) { if ($shapeLoop.Name -eq '反馈回路') { $returnLinesLoop += $shapeLoop } }
if ($returnLinesLoop.Count -ne 3) { throw 'Unexpected feedback connectors' }
Set-BoxLoop $returnLinesLoop[0] 7.90 5.10 0 1.15
Set-BoxLoop $returnLinesLoop[1] 2.10 6.25 5.80 0
Set-BoxLoop $returnLinesLoop[2] 2.10 5.10 0 1.15
$feedbackTextLoop = $slideLoop.Shapes.Item('分析反馈，修改算子')
Set-BoxLoop $feedbackTextLoop 3.40 5.72 4.00 .40
$feedbackTextLoop.TextFrame.TextRange.Font.Size = 20

# Split the existing label so only the simulator receives emphasis.
$oldLabelLoop = $null
foreach ($shapeLoop in $slideLoop.Shapes) { if ($shapeLoop.Name -eq '真机 / 模拟器') { $oldLabelLoop = $shapeLoop } }
if ($null -ne $oldLabelLoop) { $oldLabelLoop.Delete() }
for ($labelIndexLoop=$slideLoop.Shapes.Count;$labelIndexLoop -ge 1;$labelIndexLoop--) {
    if ($slideLoop.Shapes.Item($labelIndexLoop).Name -in @('Loop真机标签','Loop模拟器标签','模拟器到时序仿真箭头')) { $slideLoop.Shapes.Item($labelIndexLoop).Delete() }
}
$machineLabelLoop = New-TextLoop $slideLoop 'Loop真机标签' '真机 /' 3.86 4.02 1.03 .40 20 $mutedLoop
$simulatorLabelLoop = New-TextLoop $slideLoop 'Loop模拟器标签' '模拟器' 4.91 4.02 1.10 .40 20 $mutedLoop

# Preserve all original table text and row colors; only resize and reflow.
$tableShapeLoop = $null
foreach ($shapeLoop in $slideLoop.Shapes) { if ($shapeLoop.HasTable -eq -1) { $tableShapeLoop=$shapeLoop; break } }
if ($null -eq $tableShapeLoop) { throw 'Missing simulator classification table' }
$tableShapeLoop.Name = '模拟器分类表'
$tableShapeLoop.Left = [single](7.50*72)
$tableShapeLoop.Top = [single](1.35*72)
$tableLoop = $tableShapeLoop.Table
$columnWidthsLoop = @(1.28, 1.57, 2.40)
for ($colLoop=1; $colLoop -le 3; $colLoop++) { $tableLoop.Columns.Item($colLoop).Width = [single]($columnWidthsLoop[$colLoop-1] * 72) }
for ($rowLoop=1; $rowLoop -le 5; $rowLoop++) {
    for ($colLoop=1; $colLoop -le 3; $colLoop++) {
        $cellFrameLoop = $tableLoop.Cell($rowLoop,$colLoop).Shape.TextFrame
        $cellFrameLoop.MarginLeft = $cellFrameLoop.MarginRight = 5
        $cellFrameLoop.MarginTop = $cellFrameLoop.MarginBottom = 3
        $cellFrameLoop.VerticalAnchor = 3
        $cellFrameLoop.TextRange.Font.Size = 12.5
        $cellFrameLoop.TextRange.ParagraphFormat.SpaceBefore = 0
        $cellFrameLoop.TextRange.ParagraphFormat.SpaceAfter = 0
    }
    $tableLoop.Rows.Item($rowLoop).Height = [single]($(if($rowLoop -eq 1){.36}else{.46})*72)
}

# The callout points to the timing-simulation row; it remains editable.
$calloutLoop = $slideLoop.Shapes.AddLine([single](5.48*72),[single](3.96*72),[single](7.44*72),[single](2.86*72))
$calloutLoop.Name = '模拟器到时序仿真箭头'
$calloutLoop.Line.ForeColor.RGB = $navyLoop
$calloutLoop.Line.Weight = 2.4
$calloutLoop.Line.EndArrowheadStyle = 2
$calloutLoop.Line.EndArrowheadWidth = 3
$calloutLoop.Line.EndArrowheadLength = 3
$calloutLoop.Shadow.Visible = 0

# Stage 1: loop initially visible. First click: emphasis, then callout.
# Stage 3 uses its own click so the speaker controls when the table appears.
$sequenceLoop = $slideLoop.TimeLine.MainSequence
while ($sequenceLoop.Count -gt 0) { $sequenceLoop.Item(1).Delete() }
$boldLoop = $sequenceLoop.AddEffect($simulatorLabelLoop, 58, 0, 1)
# PowerPoint's ChangeFontStyle preset already sets fontWeight=bold.
$boldLoop.Timing.Duration = .25
$colorLoop = $sequenceLoop.AddEffect($simulatorLabelLoop, 56, 0, 2)
$colorLoop.EffectParameters.Color2.RGB = $navyLoop
$colorLoop.Timing.Duration = .25
$arrowLoop = $sequenceLoop.AddEffect($calloutLoop, 22, 0, 3)
$arrowLoop.EffectParameters.Direction = 4
$arrowLoop.Timing.Duration = .40
$arrowLoop.Timing.TriggerDelayTime = .10
$tableEffectLoop = $sequenceLoop.AddEffect($tableShapeLoop, 10, 0, 1)
$tableEffectLoop.Timing.Duration = .45

$animationNotesLoop = "第二页动画：进入本页先展示算子优化loop。第一次单击，模拟器标签加粗并变为深蓝，随后出现指向右上角时序仿真分类的箭头。第二次单击，小表格整体淡入。表格原文字与分类判定保留。"
foreach ($notesShapeLoop in $slideLoop.NotesPage.Shapes.Placeholders) {
    if ($notesShapeLoop.PlaceholderFormat.Type -eq 2) {
        $notesTextLoop = $notesShapeLoop.TextFrame.TextRange
        if (-not $notesTextLoop.Text.Contains($animationNotesLoop)) { $notesTextLoop.Text = $notesTextLoop.Text.TrimEnd() + "`r`r" + $animationNotesLoop }
        break
    }
}
$deckLoop.Save()
$slideLoop.Export((Join-Path $reportLoop 'slide2-after.png'),'PNG',2560,1440)
$overflowLoop = @()
foreach ($shapeLoop in $slideLoop.Shapes) {
    if ($shapeLoop.HasTextFrame -eq -1 -and $shapeLoop.TextFrame.HasText -eq -1) {
        if ($shapeLoop.TextFrame2.TextRange.BoundWidth -gt $shapeLoop.Width+1 -or $shapeLoop.TextFrame2.TextRange.BoundHeight -gt $shapeLoop.Height+1) { $overflowLoop += $shapeLoop.Name }
    }
    if ($shapeLoop.HasTable -eq -1) {
        for ($rowLoop=1;$rowLoop -le $shapeLoop.Table.Rows.Count;$rowLoop++) {
            for ($colLoop=1;$colLoop -le $shapeLoop.Table.Columns.Count;$colLoop++) {
                $cellLoop=$shapeLoop.Table.Cell($rowLoop,$colLoop).Shape
                if ($cellLoop.TextFrame2.TextRange.BoundWidth -gt $cellLoop.Width+1 -or $cellLoop.TextFrame2.TextRange.BoundHeight -gt $cellLoop.Height+1) { $overflowLoop += "cell-$rowLoop-$colLoop" }
            }
        }
    }
}
$otherPageChecksLoop = @()
foreach ($otherLoop in $deckLoop.Slides) {
    if ($otherLoop.SlideIndex -ne 2) {
        $afterPathLoop = Join-Path $reportLoop ('after-slide-' + $otherLoop.SlideIndex + '.png')
        $beforePathLoop = Join-Path $reportLoop ('before-slide-' + $otherLoop.SlideIndex + '.png')
        $otherLoop.Export($afterPathLoop,'PNG',1600,900)
        $otherPageChecksLoop += @{ slide=$otherLoop.SlideIndex; rendering_unchanged=((Get-FileHash -LiteralPath $afterPathLoop).Hash -eq (Get-FileHash -LiteralPath $beforePathLoop).Hash) }
    }
}
$animationLoop = @(foreach ($effectLoop in $sequenceLoop) { @{ type=[int]$effectLoop.EffectType; shape=$effectLoop.Shape.Name; trigger=[int]$effectLoop.Timing.TriggerType; duration=$effectLoop.Timing.Duration } })
$resultLoop = @{ main=$MainPresentation; slides=$deckLoop.Slides.Count; backup_directory=$backupLoop; text_overflows=@($overflowLoop); other_slides=$otherPageChecksLoop; animations=$animationLoop; native_editable_table=$true; slide2_only=$true; status='saved' }
$resultLoop | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $reportLoop 'validation.json') -Encoding UTF8
$resultLoop | ConvertTo-Json -Depth 10
if ($overflowLoop.Count -gt 0 -or @($otherPageChecksLoop | Where-Object { -not $_.rendering_unchanged }).Count -gt 0) { throw 'Slide validation failed' }

$previewDirectoryLoop = Join-Path $deliveryLoop 'material/01_开场与总览'
Copy-Item -LiteralPath (Join-Path $reportLoop 'slide2-after.png') -Destination (Join-Path $previewDirectoryLoop '主PPT-第二页-loop.png') -Force
$animationNotesLoop | Set-Content -LiteralPath (Join-Path $previewDirectoryLoop '主PPT-第二页-动画说明.md') -Encoding UTF8
$deckLoop.Windows.Item(1).View.GotoSlide(2)
