param([string]$MainPresentation='E:\0_XSAI\00_KernelAgent\0_HGKDA_docs\BAAI-reports\06_work\智源20261124汇报.pptx')
$ErrorActionPreference='Stop'
$OutputEncoding=[Console]::OutputEncoding=[Text.UTF8Encoding]::new()
$reportTask=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../reports/main-technical-animation-20261123'))
$workTask=Split-Path -Parent $MainPresentation
$backupTask=Join-Path $workTask 'material/archive/20261123_技术页动画调整前'
New-Item -ItemType Directory -Path $reportTask -Force | Out-Null
New-Item -ItemType Directory -Path $backupTask -Force | Out-Null
$stampTask=[TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,'China Standard Time').ToString('yyyyMMdd_HHmmss')
Copy-Item -LiteralPath $MainPresentation -Destination (Join-Path $backupTask ($stampTask+'_磁盘原件.pptx'))

function Get-ShapeTask($slideTask,[int]$idTask){
    foreach($shapeTask in $slideTask.Shapes){if($shapeTask.Id -eq $idTask){return $shapeTask}}
    throw "Missing shape $idTask"
}
function Add-StageTask($slideTask,[string]$labelTask,[int[]]$idsTask){
    $firstTask=$true
    foreach($idTask in $idsTask){
        $shapeTask=Get-ShapeTask $slideTask $idTask
        $triggerTask=if($firstTask){1}else{2}
        $effectTask=$slideTask.TimeLine.MainSequence.AddEffect($shapeTask,10,0,$triggerTask)
        $effectTask.Timing.Duration=.4
        $effectTask.Timing.TriggerDelayTime=0
        $firstTask=$false
    }
    return @{label=$labelTask;shape_ids=@($idsTask);duration=.4;trigger='click'}
}

$pptTask=$null
$deckTask=$null
$openedTask=$false
try{
    try{$pptTask=[Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application')}
    catch{$pptTask=New-Object -ComObject PowerPoint.Application}
    foreach($openTask in $pptTask.Presentations){if($openTask.FullName -eq $MainPresentation){$deckTask=$openTask;break}}
    if($null -eq $deckTask){$deckTask=$pptTask.Presentations.Open($MainPresentation,0,0,0);$openedTask=$true}
    $deckTask.SaveCopyAs((Join-Path $backupTask ($stampTask+'_编辑前快照.pptx')),24)
    if((Get-ShapeTask $deckTask.Slides.Item(6) 2).TextFrame.TextRange.Text -notlike '*成对 profiling*'){throw 'Unexpected slide 6'}
    if((Get-ShapeTask $deckTask.Slides.Item(7) 2).TextFrame.TextRange.Text -notlike '*模拟器分析*'){throw 'Unexpected slide 7'}
    foreach($slideTask in $deckTask.Slides){
        $slideTask.Export((Join-Path $reportTask ('before-'+$slideTask.SlideIndex+'.png')),'PNG',1600,900)
    }
    foreach($indexTask in @(6,7)){if($deckTask.Slides.Item($indexTask).TimeLine.MainSequence.Count -ne 0){throw "Slide $indexTask already has animations"}}
    $slideTask=$deckTask.Slides.Item(6)
    $profileStagesTask=@(
        (Add-StageTask $slideTask '成对采集' @(5,6,7,8,9,29,30,31))
        (Add-StageTask $slideTask '两组指标对比' @(3,4,10,11,12,13,32,33,34,35,36,37,38))
        (Add-StageTask $slideTask '差距标准化' @(14,15,16,17,39,40,41))
        (Add-StageTask $slideTask 'Analyst / Coder Agent' @(18,42,43,44))
        (Add-StageTask $slideTask '编译验证' @(19,45,46))
        (Add-StageTask $slideTask '反馈迭代' @(20,21,22,23,24))
    )
    $slideTask=$deckTask.Slides.Item(7)
    $simStagesTask=@(
        (Add-StageTask $slideTask 'Inst duration' @(3,5,7,8,9,30,31))
        (Add-StageTask $slideTask 'Top-K / 编译调用栈 / AST' @(12,32,33,34,35))
        (Add-StageTask $slideTask '热点区域' @(13,14,15,36,37))
        (Add-StageTask $slideTask '执行 trace' @(4,6,10,11,38,39))
        (Add-StageTask $slideTask '有效执行 / 等待 / 空泡 / 重叠' @(16,40,41))
        (Add-StageTask $slideTask 'Analyst / Coder Agent' @(17,18,19,20,21,43,44,49,51,52))
        (Add-StageTask $slideTask '编译验证' @(22,50,45,46))
        (Add-StageTask $slideTask '反馈迭代' @(23,24,25,26))
    )
    foreach($indexTask in @(6,7)){
        $slideTask=$deckTask.Slides.Item($indexTask)
        $stagesTask=if($indexTask -eq 6){$profileStagesTask}else{$simStagesTask}
        $notesTask='分步动画：进页显示标题和左侧输入；每次点击展开下一步，文字、方框和连接箭头同步淡入，时长0.4秒。顺序：'+(($stagesTask|ForEach-Object{$_.label}) -join ' → ')+'。先前已出现的步骤保留，最后呈现完整流程。'
        foreach($noteShapeTask in $slideTask.NotesPage.Shapes.Placeholders){
            if($noteShapeTask.PlaceholderFormat.Type -eq 2){$noteShapeTask.TextFrame.TextRange.Text=$noteShapeTask.TextFrame.TextRange.Text.TrimEnd()+"`r`r"+$notesTask;break}
        }
        $slideTask.SlideShowTransition.AdvanceOnTime=0
        $slideTask.SlideShowTransition.AdvanceOnClick=-1
    }
    $deckTask.Save()
    $checksTask=@()
    foreach($slideTask in $deckTask.Slides){
        $afterTask=Join-Path $reportTask ('after-'+$slideTask.SlideIndex+'.png')
        $slideTask.Export($afterTask,'PNG',1600,900)
        $checksTask+=@{slide=$slideTask.SlideIndex;render_unchanged=((Get-FileHash -LiteralPath $afterTask).Hash -eq (Get-FileHash -LiteralPath (Join-Path $reportTask ('before-'+$slideTask.SlideIndex+'.png'))).Hash)}
    }
    $resultTask=@{main=$MainPresentation;slides=$deckTask.Slides.Count;backup=$backupTask;profile_stages=$profileStagesTask;simulator_stages=$simStagesTask;render_checks=$checksTask;status='saved'}
    $resultTask|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $reportTask 'validation.json') -Encoding UTF8
    $resultTask|ConvertTo-Json -Depth 10
    if(@($checksTask|Where-Object{-not $_.render_unchanged}).Count -gt 0){throw 'Unexpected visual change'}
}finally{
    if($openedTask -and $null -ne $deckTask){$deckTask.Close()}
    if($null -ne $pptTask){[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($pptTask)}
}
