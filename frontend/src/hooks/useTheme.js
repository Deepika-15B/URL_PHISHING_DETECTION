import {useEffect,useState} from 'react'; export function useTheme(){const [dark,setDark]=useState(()=>localStorage.theme!=='light');useEffect(()=>{document.documentElement.classList.toggle('dark',dark);localStorage.theme=dark?'dark':'light'},[dark]);return[dark,setDark]}

