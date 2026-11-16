#define AppName "ironasu Studio"
#define AppVersion "1.9.0"
#define AppExeName "ironasu-studio.exe"

[Setup]
AppId={{79A0B2C3-D4E5-467F-89A0-B123456789AB}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=ironasu contributors
AppPublisherURL=https://github.com/kobayashi-cyder/ironasu
SetupIconFile=..\..\dist\ironasu.ico
DefaultDirName={localappdata}\Programs\ironasu Studio
DefaultGroupName=ironasu Studio
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExeName}
OutputDir=..\..\dist\installers
OutputBaseFilename=ironasu-studio-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\..\dist\ironasu-studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\ironasu Studio"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\ironasu Studio"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch ironasu Studio"; Flags: nowait postinstall skipifsilent
