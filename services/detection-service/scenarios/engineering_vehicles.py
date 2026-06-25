"""Engineering vehicle detection scenario."""

from __future__ import annotations

from dataclasses import dataclass


VEHICLE_CLASSES = {
    "wajueji": "挖掘机",
    "chanche": "铲车",
    "dazhuangji": "打桩机",
    "yaluji": "压路机",
    "diaoche": "吊车",
    "gaokongche": "高空车",
    "youguanche": "油罐车",
    "yunshuche": "运输车",
    "other": "其他",
}


CLASS_IDS = {name: idx for idx, name in enumerate(VEHICLE_CLASSES)}


VEHICLE_CLASS_FEATURES = """类别定位特征：
wajueji/挖掘机：履带或轮式底盘，上车可360°回转，最显著特征为多关节细长挖臂（动臂＋斗杆）末端挂载内弯铲斗，作业时弯臂、斗口朝向地面。区别于吊车：末端为铲斗而非吊钩，无柔性钢丝绳；区别于铲车：挖臂细长多节可大幅折叠，铲斗可内旋入土。
chanche/铲车：轮式装载机，前端配一对粗短举升臂驱动大容量开口铲斗，车身宽大圆润，以四轮驱动为主。铲斗只做垂直升降，无折点和回转结构，以此区别于挖掘机；前端为铲斗而无吊钩、钢丝绳或支腿，以此区别于吊车。
dazhuangji/打桩机：核心特征为一根极其高耸、粗壮、近乎绝对竖直（≈90°）的刚性桅杆，轨道上挂载旋挖钻杆、液压动力头或重型桩锤等桩工装备。与吊车的关键区别：桅杆刚性固定、不可变幅旋转，且有明确入地的桩工部件。
yaluji/压路机：整机低矮扁平，前后配有大直径光面钢轮（振动式）或宽排充气轮胎（轮胎式），专用于路面碾压压实。无任何臂、斗、罐体或竖向高耸结构，标志性钢轮或宽压实轮胎一旦清晰可见即可确认。
diaoche/吊车：主要用于吊装重物。典型特征是起重底盘或履带底盘，带有较长的可变幅起重臂，并通过柔性钢丝绳连接吊钩或吊具。车辆可能带有支腿，也可能是履带式结构。与打桩机的区别：依赖柔性钢丝绳悬挂、臂杆可俯仰变幅，无竖直入地的桩工配件；
gaokongche/高空车：主要用于人员高空作业。典型特征是车辆或底盘上安装伸缩臂、折叠臂或曲臂，臂的末端有供人员站立的封闭或半封闭作业平台，也可称为吊篮、工作斗。区别于吊车：末端为载人平台而非吊钩，无重型起重能力；区别于打桩机：作业臂可多向弯折且末端平台朝上，而非刚性竖杆入地。
youguanche/油罐车：主要用于运输油料、液体或类似流体物质。典型特征是卡车后部装有封闭的圆柱形或椭圆形金属罐体，罐体通常较光滑，两端为弧形封头，底部或侧面可能有管道、阀门、接口等结构。与运输车区别：油罐车的后部不是开放货斗、普通货厢、自卸斗或搅拌筒，而是封闭液体罐体。
yunshuche/运输车：主要用于运输土方、砂石、建筑材料、混凝土、设备或普通货物。典型结构是卡车底盘，后部有开放货斗、密封货厢、自卸翻斗、平板载货区，或用于运输混凝土的旋转搅拌筒。无专用作业臂、铲斗、压实钢轮或封闭圆形罐体。
other/其他：适用于两种情形：目标确属工程机械或施工车辆但不符合上述任何类别的核心特征（如摊铺机、平地机、泵车、清扫车等）；或关键部件因遮挡、角度、画质无法确认，类别证据严重不足。不得作为默认兜底滥用，应有明确归入理由。
"""


@dataclass(frozen=True)
class DetectionScenario:
    """Static metadata for one detection scenario."""

    name: str
    classes: dict[str, str]
    class_features: str


SCENARIO = DetectionScenario(
    name="engineering_vehicles",
    classes=VEHICLE_CLASSES,
    class_features=VEHICLE_CLASS_FEATURES,
)
